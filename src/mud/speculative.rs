use crate::mud::slime::SlimeWorkspace;
use crate::mud::slime_forward::SlimeLayer;
use crate::mud::slime_forward::evaluate_slime_block;

/// DSpark Speculative Drafter (Priority 39)
/// Proposes K token candidates using a lightweight ternary model (e.g. first 2 layers).
pub struct SlimeDrafter<'a> {
    layers: Vec<&'a SlimeLayer>,
    output_weight: *const f32,
    hidden: usize,
    vocab_size: usize,
    // P-01: Zero-allocation buffers
    regs_buf: Vec<f32>,
    logits_buf: Vec<f32>,
}

impl<'a> SlimeDrafter<'a> {
    pub fn new(
        all_layers: &'a [SlimeLayer],
        num_draft_layers: usize,
        output_weight: *const f32,
        hidden: usize,
        vocab_size: usize,
    ) -> Self {
        let layers: Vec<&'a SlimeLayer> = all_layers
            .iter()
            .take(num_draft_layers)
            .collect();

        Self {
            layers,
            output_weight,
            hidden,
            vocab_size,
            regs_buf: vec![0.0; hidden],
            logits_buf: vec![0.0; vocab_size],
        }
    }

    /// Proposes K tokens.
    /// The drafter uses the main workspace, temporarily advancing the KV cache.
    /// The main model will overwrite these positions during verification.
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn propose_tokens(
        &mut self,
        ws: &mut SlimeWorkspace,
        start_pos: usize,
        k: usize,
        eps: f32,
        current_token: u32,
        token_embd_ptr: *const f32,
    ) -> Vec<u32> {
        let mut proposals = Vec::with_capacity(k);
        let mut tok = current_token;
        
        if self.output_weight.is_null() || token_embd_ptr.is_null() {
            return proposals;
        }
        
        let pool = crate::mud::pcore_pool::get_pool();
        let rows_per_task = ((self.vocab_size / 8) / 4 * 4).max(4);
        let out_w_p = self.output_weight as usize;
        
        for step in 0..k {
            let pos = start_pos + step;
            
            // 1. Embed token
            let row_start = tok as usize * self.hidden;
            for h in 0..self.hidden {
                let emb_val = unsafe { *token_embd_ptr.add(row_start + h) };
                crate::mud::slime::SlimeRegister::init_from_embed(&mut ws.registers[h], &mut ws.jepa_z, h, ws.hidden_size, ws.num_layers, emb_val, pos == 0);
            }
            
            // 2. Run draft layers
            for (l_idx, layer) in self.layers.iter().enumerate() {
                evaluate_slime_block(layer, l_idx, ws, pos, eps, None);
            }
            
            // 3. Simple Greedy LM Head (Parallelized & Zero-Allocation)
            for h in 0..self.hidden {
                self.regs_buf[h] = ws.registers[h].read_accum();
            }
                
            let regs_p = self.regs_buf.as_ptr() as usize;
            let logits_p = self.logits_buf.as_mut_ptr() as usize;
            let hidden_dim = self.hidden;
            let vocab = self.vocab_size;
            
            for i in 0..8 {
                let start_row = i * rows_per_task;
                let end_row = if i == 7 { vocab } else { start_row + rows_per_task };
                if start_row >= end_row { break; }
                
                pool.execute(move || {
                    let r_ptr = regs_p as *const f32;
                    let l_ptr = logits_p as *mut f32;
                    let w_ptr = out_w_p as *const f32;
                    let rows = end_row - start_row;
                    
                    unsafe {
                        // Compute sgemm for a subset of the vocabulary (subset of rows of output_weight)
                        // sgemm_abt computes: out[i] = dot(A[0], B[i]) 
                        // A is 1xHidden, B is VocabXHidden.
                        crate::asm::sgemm_abt(1, rows, hidden_dim, r_ptr, w_ptr.add(start_row * hidden_dim), l_ptr.add(start_row));
                    }
                });
            }
            pool.wait_all();
            
            // Find max logit
            let mut best_id = 0;
            let mut max_val = f32::NEG_INFINITY;
            for (i, &l) in self.logits_buf.iter().enumerate() {
                if l > max_val {
                    max_val = l;
                    best_id = i;
                }
            }
            
            proposals.push(best_id as u32);
            tok = best_id as u32;
        }
        
        proposals
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drafter_initialization() {
        // Just verify we can instantiate it cleanly
        let layers = vec![];
        let drafter = SlimeDrafter::new(&layers, 2, std::ptr::null(), 256, 1000);
        assert_eq!(drafter.layers.len(), 0);
    }
}
