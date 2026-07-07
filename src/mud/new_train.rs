    fn train_on_sequence(
        &self,
        mud: &mut MudFile,
        shadow_emb: &mut [f32],
        layers: &[crate::mud::slime_forward::SlimeLayer],
        shadow_layers: &mut [crate::mud::SlimeLayerShadowF32],
        workspace: &mut crate::mud::slime::SlimeWorkspace,
        backward_ws: &mut crate::mud::slime_backward::SlimeBackwardWorkspace,
        tapes: &mut [forge_autograd::Tape],
        gradients: &mut [crate::mud::slime_backward::SlimeLayerGradients],
        tokens: &[u32],
        batch_size: usize,
    ) -> anyhow::Result<f32> {
        let lr = crate::mud::constants::QAT_LEARNING_RATE;
        let hidden_size = workspace.hidden_size;
        let vocab_size = shadow_emb.len() / hidden_size;

        let pairs: Vec<(usize, usize)> = tokens
            .windows(2)
            .step_by(8)
            .take(batch_size)
            .filter_map(|w| {
                let inp = w[0] as usize;
                let tgt = w[1] as usize;
                if inp < vocab_size && tgt < vocab_size {
                    Some((inp, tgt))
                } else {
                    None
                }
            })
            .collect();

        let mut total_loss = 0.0f32;
        let mut pair_count = 0;
        let eps = 1e-6; // rms norm eps

        for (input_id, target_id) in pairs {
            if crate::mud::corpus_trainer::SHOULD_TERMINATE.load(std::sync::atomic::Ordering::SeqCst) { break; }
            
            workspace.clear_registers();
            for t in tapes.iter_mut() { t.reset(); }
            for g in gradients.iter_mut() { g.reset(); }
            
            // 1. Load embedding
            let emb_offset = input_id * hidden_size;
            let mut x_data = shadow_emb[emb_offset..emb_offset + hidden_size].to_vec();
            let absmean_x = x_data.iter().map(|v| v.abs()).sum::<f32>() / hidden_size as f32;
            let scale_x = (absmean_x * 0.707).max(1e-8);
            for v in &mut x_data {
                *v = (*v / scale_x).round().clamp(-1.0, 1.0) * scale_x;
            }
            
            for i in 0..hidden_size {
                crate::mud::slime::SlimeRegister::init_from_embed(
                    &mut workspace.registers[i],
                    &mut workspace.jepa_z,
                    i,
                    hidden_size,
                    layers.len(),
                    x_data[i],
                    true
                );
            }
            
            // 2. Forward pass through layers
            for (l_idx, layer) in layers.iter().enumerate() {
                crate::mud::slime_forward::evaluate_slime_block(layer, l_idx, workspace, 0, eps, Some(&mut tapes[l_idx]));
            }
            
            let mut final_x = vec![0.0f32; hidden_size];
            for i in 0..hidden_size {
                final_x[i] = workspace.registers[i].read_accum();
            }

            // 3. Contrastive Logits against Vocabulary
            const NUM_NEG: usize = 7;
            let mut rng_state = input_id.wrapping_mul(1664525).wrapping_add(target_id);
            let mut neg_ids: Vec<usize> = Vec::with_capacity(NUM_NEG);
            while neg_ids.len() < NUM_NEG {
                rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
                let neg = rng_state % vocab_size;
                if neg != target_id && !neg_ids.contains(&neg) {
                    neg_ids.push(neg);
                }
            }
            
            let num_classes = 1 + neg_ids.len();
            let mut class_embs = Vec::with_capacity(num_classes * hidden_size);
            
            // Positive target
            {
                let start = target_id * hidden_size;
                class_embs.extend_from_slice(&shadow_emb[start..start + hidden_size]);
                let slice = &mut class_embs[0..hidden_size];
                let absmean = slice.iter().map(|v| v.abs()).sum::<f32>() / (hidden_size as f32);
                let scale = (absmean * 0.707).max(1e-8);
                for v in slice { *v = (*v / scale).round().clamp(-1.0, 1.0) * scale; }
            }
            // Negative targets
            for (ni, &neg) in neg_ids.iter().enumerate() {
                let start = neg * hidden_size;
                class_embs.extend_from_slice(&shadow_emb[start..start + hidden_size]);
                let slice = &mut class_embs[(1 + ni) * hidden_size..(2 + ni) * hidden_size];
                let absmean = slice.iter().map(|v| v.abs()).sum::<f32>() / (hidden_size as f32);
                let scale = (absmean * 0.707).max(1e-8);
                for v in slice { *v = (*v / scale).round().clamp(-1.0, 1.0) * scale; }
            }
            
            let mut tape = forge_autograd::Tape::new();
            let x_node = tape.push_leaf(final_x, vec![1, hidden_size]);
            let emb_node = tape.push_leaf(class_embs, vec![num_classes, hidden_size]);
            let logits = tape.linear(x_node, emb_node);
            let loss = tape.cross_entropy(logits, 0);
            tape.backward(loss);
            
            total_loss += tape.nodes[loss.0].data[0];
            pair_count += 1;
            
            // 4. Backpropagate dx through 30 layers
            let mut grad_in = tape.nodes[x_node.0].grad.clone();
            let mut grad_out = vec![0.0f32; hidden_size];
            
            if grad_in.iter().all(|v| v.is_finite()) {
                let norm_sq: f32 = grad_in.iter().map(|&g| g * g).sum();
                let clip = if norm_sq.sqrt() > 1.0 { 1.0 / norm_sq.sqrt() } else { 1.0 };
                for v in &mut grad_in { *v *= clip; }
                
                for (l_idx, layer) in layers.iter().enumerate().rev() {
                    crate::mud::slime_backward::backward_slime_block(
                        layer,
                        workspace,
                        backward_ws,
                        &tapes[l_idx],
                        &mut gradients[l_idx],
                        &grad_in,
                        &mut grad_out
                    );
                    grad_in.copy_from_slice(&grad_out);
                }
                
                // grad_in now contains the gradient with respect to the input embeddings
                // Apply learning rate and update shadow_emb
                let target_slice = &mut shadow_emb[input_id * hidden_size..(input_id + 1) * hidden_size];
                unsafe { forge_autograd::avx_math::axpy_avx2(target_slice, -lr, &grad_in); }
            }
            
            // 5. Update target and negative embeddings
            for node in tape.nodes.iter().filter(|n| matches!(n.op, forge_autograd::Op::Leaf)) {
                if node.shape.len() == 2 && node.shape[0] == num_classes && node.shape[1] == hidden_size {
                    let demb = &node.grad;
                    let target_grad = &demb[0..hidden_size];
                    let target_row = &mut shadow_emb[target_id * hidden_size..(target_id + 1) * hidden_size];
                    if target_grad.iter().all(|v| v.is_finite()) {
                        let norm_sq: f32 = target_grad.iter().map(|&g| g * g).sum();
                        let clip = if norm_sq.sqrt() > 1.0 { 1.0 / norm_sq.sqrt() } else { 1.0 };
                        unsafe { forge_autograd::avx_math::axpy_avx2(target_row, -lr * clip, target_grad); }
                    }
                    for (ni, &neg_id) in neg_ids.iter().enumerate() {
                        let neg_grad = &demb[(1 + ni) * hidden_size..(2 + ni) * hidden_size];
                        if neg_grad.iter().all(|v| v.is_finite()) {
                            let norm_sq: f32 = neg_grad.iter().map(|&g| g * g).sum();
                            let clip = if norm_sq.sqrt() > 1.0 { 1.0 / norm_sq.sqrt() } else { 1.0 };
                            let neg_row = &mut shadow_emb[neg_id * hidden_size..(neg_id + 1) * hidden_size];
                            unsafe { forge_autograd::avx_math::axpy_avx2(neg_row, -lr * clip, neg_grad); }
                        }
                    }
                    break;
                }
            }
        }
        
        // 6. Aggregate gradients into shadow_layers
        if pair_count > 0 {
            for (l_idx, shadow_layer) in shadow_layers.iter_mut().enumerate() {
                let grad = &gradients[l_idx];
                for (i, v) in shadow_layer.q_w.iter_mut().enumerate() { *v -= lr * grad.q_w_grad[i]; }
                for (i, v) in shadow_layer.k_w.iter_mut().enumerate() { *v -= lr * grad.k_w_grad[i]; }
                for (i, v) in shadow_layer.v_w.iter_mut().enumerate() { *v -= lr * grad.v_w_grad[i]; }
                for (i, v) in shadow_layer.o_w.iter_mut().enumerate() { *v -= lr * grad.o_w_grad[i]; }
                for (i, v) in shadow_layer.ffn_up_w.iter_mut().enumerate() { *v -= lr * grad.ffn_up_w_grad[i]; }
                for (i, v) in shadow_layer.ffn_gate_w.iter_mut().enumerate() { *v -= lr * grad.ffn_gate_w_grad[i]; }
                for (i, v) in shadow_layer.ffn_down_w.iter_mut().enumerate() { *v -= lr * grad.ffn_down_w_grad[i]; }
            }
        }

        if pair_count > 0 {
            Ok(total_loss / pair_count as f32)
        } else {
            Ok(0.0)
        }
    }
