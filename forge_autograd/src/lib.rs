pub mod avx_math;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeId(pub usize);

#[derive(Clone, Debug)]
pub enum Op {
    Leaf,
    /// z = x + y (elemento a elemento)
    Add(NodeId, NodeId),
    /// Multiplicación matricial (Linear Layer): Z = X * W^T
    /// X tiene forma [M, K], W tiene forma [N, K]. Z tiene forma [M, N].
    MatMul(NodeId, NodeId),
    /// Activación SiLU (x * sigmoid(x))
    SiLU(NodeId),
    /// Multiplicación elemento a elemento: Z = X * Y
    Mul(NodeId, NodeId),
    /// CrossEntropyLoss(Logits, Target_Index) -> Scalar Loss
    CrossEntropy(NodeId, usize),
    /// STE Quantization: forward = round(x / s).clamp(-1, 1) * s, backward = dL/dx (identity)
    STEQuantize(NodeId, NodeId),
    /// RMSNorm: y = x * rms_norm_scale(x, eps) * w, backward = proper gradient
    RMSNorm(NodeId, NodeId, f32),
    /// KL Divergence: KL(student_logits || teacher_logits) with temperature
    KLDiv(NodeId, NodeId, f32),
    /// Softmax a lo largo del último eje. Input: [M, N] o [N]. Output mismas dimensiones.
    Softmax(NodeId),
    /// Transposición de matriz 2D. Input: [M, N]. Output: [N, M].
    Transpose(NodeId),
    /// Reshape: output shape dada explícitamente, datos compartidos (sin copia).
    Reshape(NodeId, Vec<usize>),
    /// Multi-Head Causal Self-Attention con GQA (Grouped Query Attention).
    /// Q: [seq_len, n_head * head_dim], K/V: [seq_len, n_kv_head * head_dim], mask: [seq_len, seq_len]
    /// Output: [seq_len, n_head * head_dim]
    MultiHeadAttention {
        q: NodeId,
        k: NodeId,
        v: NodeId,
        mask: NodeId,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
    },
    /// VICReg: Variance-Invariance-Covariance Regularization (TRAIN-03).
    /// Input: [seq_len, hidden]. Output: scalar loss.
    /// Aplica regularización de varianza y covarianza para prevenir colapso de la representación.
    VICReg(NodeId, f32),
    /// SelectRow: extrae una fila de un tensor 2D [N, D] → [1, D].
    SelectRow(NodeId, usize),
    /// KLDivDirect: KL(p||q) sin softmax. p,q son distribuciones de probabilidad [N].
    /// loss = sum_i p_i * ln(p_i / q_i). backward: d(loss)/dp_i = ln(p_i/q_i) + 1
    KLDivDirect(NodeId, NodeId),
    /// TernaryLinear: Z = X * W_ternary^T * diag(scales) for frozen layers.
    /// No FP32 dequant. packed_w: [n_out * ceil(n_in/16)] u32. scales: [n_out] f32.
    TernaryLinear {
        x: NodeId,
        packed_w: Vec<u32>,
        scales: Vec<f32>,
        n_in: usize,
        n_out: usize,
    },
}

#[derive(Clone, Debug)]
pub struct Node {
    pub data: std::sync::Arc<Vec<f32>>,
    pub grad: Vec<f32>,
    pub shape: Vec<usize>,
    pub op: Op,
    /// Almacenamiento extra para datos intermedios (ej: pesos de atención para QAT-06).
    pub extra: Option<std::sync::Arc<Vec<f32>>>,
}

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Default, Clone)]
pub struct Tape {
    pub nodes: Vec<Node>,
    pub abort: Option<Arc<AtomicBool>>,
}

impl Tape {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            abort: None,
        }
    }

    pub fn with_abort(abort: Arc<AtomicBool>) -> Self {
        Self {
            nodes: Vec::new(),
            abort: Some(abort),
        }
    }

    fn should_abort(&self) -> bool {
        self.abort
            .as_ref()
            .map_or(false, |a| a.load(Ordering::Relaxed))
    }

    pub fn zero_grad(&mut self) {
        for node in &mut self.nodes {
            for g in &mut node.grad {
                *g = 0.0;
            }
        }
    }

    /// Limpia la cinta para reusar memoria en el siguiente token
    pub fn reset(&mut self) {
        self.nodes.clear();
    }

    /// Empuja un tensor de pesos o entrada (Hoja)
    pub fn push_leaf(&mut self, data: Vec<f32>, shape: Vec<usize>) -> NodeId {
        let len = data.len();
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            data: std::sync::Arc::new(data),
            grad: vec![0.0; len],
            shape,
            op: Op::Leaf,
            extra: None,
        });
        id
    }

    /// Empuja una hoja compartiendo el array de datos (Zero-Allocation)
    pub fn push_leaf_arc(&mut self, data: std::sync::Arc<Vec<f32>>, shape: Vec<usize>) -> NodeId {
        let len = data.len();
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            data,
            grad: vec![0.0; len],
            shape,
            op: Op::Leaf,
            extra: None,
        });
        id
    }

    /// Suma dos tensores elemento por elemento
    pub fn add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let (lhs_node, rhs_node) = self.get_two(lhs, rhs);
        assert_eq!(
            lhs_node.shape, rhs_node.shape,
            "Las formas deben coincidir para Add"
        );
        let len = lhs_node.data.len();
        let mut data = vec![0.0; len];

        unsafe {
            // z = x + y equivalente a z = 1.0 * x + y
            data.copy_from_slice(&rhs_node.data);
            avx_math::axpy_avx2(&mut data, 1.0, &lhs_node.data);
        }

        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            data: std::sync::Arc::new(data),
            grad: vec![0.0; len],
            shape: lhs_node.shape.clone(),
            op: Op::Add(lhs, rhs),
            extra: None,
        });
        id
    }

    /// Multiplica dos tensores elemento a elemento
    pub fn mul(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let (lhs_node, rhs_node) = self.get_two(lhs, rhs);
        assert_eq!(
            lhs_node.shape, rhs_node.shape,
            "Las formas deben coincidir para Mul"
        );
        let len = lhs_node.data.len();
        let mut data = vec![0.0; len];

        for i in 0..len {
            data[i] = lhs_node.data[i] * rhs_node.data[i];
        }

        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            data: std::sync::Arc::new(data),
            grad: vec![0.0; len],
            shape: lhs_node.shape.clone(),
            op: Op::Mul(lhs, rhs),
            extra: None,
        });
        id
    }

    /// Multiplica matrices Z = X * W^T
    /// X: [M, K], W: [N, K] -> Z: [M, N]
    pub fn linear(&mut self, x_id: NodeId, w_id: NodeId) -> NodeId {
        let x_node = &self.nodes[x_id.0];
        let w_node = &self.nodes[w_id.0];

        let m = x_node.shape[0];
        let k = x_node.shape[1];
        let n = w_node.shape[0];
        assert_eq!(w_node.shape[1], k, "Dimensión K debe coincidir");

        let mut z_data = vec![0.0; m * n];

        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe {
                avx_math::sgemm_abt_avx2(m, n, k, &x_node.data, &w_node.data, &mut z_data);
            }
        } else {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for p in 0..k {
                        sum += x_node.data[i * k + p] * w_node.data[j * k + p];
                    }
                    z_data[i * n + j] = sum;
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for p in 0..k {
                        sum += x_node.data[i * k + p] * w_node.data[j * k + p];
                    }
                    z_data[i * n + j] = sum;
                }
            }
        }

        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            data: std::sync::Arc::new(z_data),
            grad: vec![0.0; m * n],
            shape: vec![m, n],
            op: Op::MatMul(x_id, w_id),
            extra: None,
        });
        id
    }

    /// Frozen-layer linear: Z = X * W_ternary^T * diag(scales).
    /// packed_w: [n_out * ceil(n_in/16)] u32 (2-bit packed ternary).
    /// scales: [n_out] f32 (PRQ per-row scales).
    pub fn ternary_linear(
        &mut self,
        x_id: NodeId,
        packed_w: Vec<u32>,
        scales: Vec<f32>,
        n_in: usize,
        n_out: usize,
    ) -> NodeId {
        let x_node = &self.nodes[x_id.0];
        let m = x_node.shape[0];
        assert_eq!(x_node.shape[1], n_in);
        let blocks_per_row = n_in / 16 + if n_in % 16 != 0 { 1 } else { 0 };

        let mut z_data = vec![0.0f32; m * n_out];
        for row in 0..m {
            let x_off = row * n_in;
            for j in 0..n_out {
                let mut sum: i32 = 0;
                let w_off = j * blocks_per_row;
                for b in 0..blocks_per_row {
                    let block = packed_w[w_off + b];
                    let base = b * 16;
                    let limit = (n_in - base).min(16);
                    for bit in 0..limit {
                        let bits = (block >> (bit * 2)) & 3;
                        match bits {
                            1 => sum += x_node.data[x_off + base + bit] as i32,
                            2 => sum -= x_node.data[x_off + base + bit] as i32,
                            _ => {}
                        }
                    }
                }
                z_data[row * n_out + j] = sum as f32 * scales[j];
            }
        }

        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            data: std::sync::Arc::new(z_data),
            grad: vec![0.0; m * n_out],
            shape: vec![m, n_out],
            op: Op::TernaryLinear {
                x: x_id,
                packed_w,
                scales,
                n_in,
                n_out,
            },
            extra: None,
        });
        id
    }

    /// Activación SiLU: f(x) = x * sigmoid(x)
    pub fn silu(&mut self, x_id: NodeId) -> NodeId {
        let x_node = &self.nodes[x_id.0];
        let mut z_data = vec![0.0; x_node.data.len()];

        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            unsafe {
                avx_math::silu_avx2(&x_node.data, &mut z_data);
            }
        } else {
            for i in 0..x_node.data.len() {
                let x = x_node.data[i];
                z_data[i] = x / (1.0 + (-x).exp());
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for i in 0..x_node.data.len() {
                let x = x_node.data[i];
                z_data[i] = x / (1.0 + (-x).exp());
            }
        }

        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            data: std::sync::Arc::new(z_data),
            grad: vec![0.0; x_node.data.len()],
            shape: x_node.shape.clone(),
            op: Op::SiLU(x_id),
            extra: None,
        });
        id
    }

    /// STE Quantization: forward = round(x / s).clamp(-1, 1) * s, backward = identity
    /// x: [N], scale: scalar [1]. Escalar el scale por token/layer.
    pub fn ste_quantize(&mut self, x_id: NodeId, scale_id: NodeId) -> NodeId {
        let x_node = &self.nodes[x_id.0];
        let scale_node = &self.nodes[scale_id.0];
        assert_eq!(scale_node.data.len(), 1, "Scale debe ser escalar [1]");
        let s = scale_node.data[0].max(1e-8);

        let mut z_data = vec![0.0; x_node.data.len()];
        for i in 0..x_node.data.len() {
            let q = (x_node.data[i] / s).round().clamp(-1.0, 1.0);
            z_data[i] = q * s;
        }

        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            data: std::sync::Arc::new(z_data),
            grad: vec![0.0; x_node.data.len()],
            shape: x_node.shape.clone(),
            op: Op::STEQuantize(x_id, scale_id),
            extra: None,
        });
        id
    }

    /// RMSNorm: y = x * rms_norm_scale(x, eps) * w
    /// x: [N] o [M, N], w: [N] (pesos aprendibles), eps: f32
    pub fn rms_norm(&mut self, x_id: NodeId, w_id: NodeId, eps: f32) -> NodeId {
        let x_node = &self.nodes[x_id.0];
        let w_node = &self.nodes[w_id.0];
        let n = w_node.data.len();
        let total_elements = x_node.data.len();
        assert_eq!(
            total_elements % n,
            0,
            "RMSNorm input debe ser múltiplo de la dimensión del peso"
        );

        let m = total_elements / n;
        let mut z_data = vec![0.0; total_elements];

        #[cfg(target_arch = "x86_64")]
        let has_avx2 = std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma");

        for i in 0..m {
            let row_start = i * n;
            let row = &x_node.data[row_start..row_start + n];

            let rms_scale = {
                #[cfg(target_arch = "x86_64")]
                {
                    if has_avx2 {
                        unsafe { avx_math::rms_norm_scale_avx2(row, eps) }
                    } else {
                        let sum_sq: f32 = row.iter().map(|v| v * v).sum();
                        1.0 / ((sum_sq / n as f32) + eps).sqrt()
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    let sum_sq: f32 = row.iter().map(|v| v * v).sum();
                    1.0 / ((sum_sq / n as f32) + eps).sqrt()
                }
            };

            for j in 0..n {
                z_data[row_start + j] = row[j] * rms_scale * w_node.data[j];
            }
        }

        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            data: std::sync::Arc::new(z_data),
            grad: vec![0.0; total_elements],
            shape: x_node.shape.clone(),
            op: Op::RMSNorm(x_id, w_id, eps),
            extra: None,
        });
        id
    }

    /// Softmax a lo largo del último eje.
    /// Input [M, N] o [N]. Output mismas dimensiones.
    pub fn softmax(&mut self, x_id: NodeId) -> NodeId {
        let x_node = &self.nodes[x_id.0];
        let len = x_node.data.len();
        let mut z_data = vec![0.0; len];

        let inner = if x_node.shape.len() >= 2 {
            x_node.shape[1]
        } else {
            x_node.shape[0]
        };
        let outer = if x_node.shape.len() >= 2 {
            x_node.shape[0]
        } else {
            1
        };

        for i in 0..outer {
            let start = i * inner;
            let end = start + inner;
            let max_val = x_node.data[start..end]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0;
            for j in start..end {
                let e = (x_node.data[j] - max_val).exp();
                z_data[j] = e;
                sum_exp += e;
            }
            for j in start..end {
                z_data[j] /= sum_exp;
            }
        }

        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            data: std::sync::Arc::new(z_data),
            grad: vec![0.0; len],
            shape: x_node.shape.clone(),
            op: Op::Softmax(x_id),
            extra: None,
        });
        id
    }

    /// Reshape: cambia la forma sin copiar datos.
    /// El número de elementos debe coincidir.
    pub fn reshape(&mut self, x_id: NodeId, new_shape: Vec<usize>) -> NodeId {
        let x_node = &self.nodes[x_id.0];
        let old_len: usize = x_node.shape.iter().product();
        let new_len: usize = new_shape.iter().product();
        assert_eq!(
            old_len, new_len,
            "Reshape: número de elementos debe coincidir"
        );
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            data: x_node.data.clone(),
            grad: vec![0.0; old_len],
            shape: new_shape,
            op: Op::Reshape(x_id, vec![]), // shape stored in node.shape
            extra: None,
        });
        id
    }

    /// Transposición de matriz 2D. Input: [M, N], Output: [N, M].
    pub fn transpose(&mut self, x_id: NodeId) -> NodeId {
        let x_node = &self.nodes[x_id.0];
        assert_eq!(x_node.shape.len(), 2, "Transpose requiere shape 2D");
        let m = x_node.shape[0];
        let n = x_node.shape[1];
        let mut z_data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                z_data[j * m + i] = x_node.data[i * n + j];
            }
        }
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            data: std::sync::Arc::new(z_data),
            grad: vec![0.0; m * n],
            shape: vec![n, m],
            op: Op::Transpose(x_id),
            extra: None,
        });
        id
    }

    /// Multi-Head Causal Self-Attention con GQA.
    pub fn mha(
        &mut self,
        q_id: NodeId,
        k_id: NodeId,
        v_id: NodeId,
        mask_id: NodeId,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
    ) -> NodeId {
        let q_node = &self.nodes[q_id.0];
        let k_node = &self.nodes[k_id.0];
        let v_node = &self.nodes[v_id.0];
        let mask_node = &self.nodes[mask_id.0];
        let seq_len = q_node.shape[0];
        assert_eq!(q_node.shape, vec![seq_len, n_head * head_dim]);
        assert_eq!(k_node.shape, vec![seq_len, n_kv_head * head_dim]);
        assert_eq!(v_node.shape, vec![seq_len, n_kv_head * head_dim]);
        assert_eq!(mask_node.shape, vec![seq_len, seq_len]);

        let repeat = n_head / n_kv_head;
        let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();

        // Reshape Q to [n_head, seq_len, head_dim]
        let mut q_3d = vec![0.0; seq_len * n_head * head_dim];
        for s in 0..seq_len {
            for h in 0..n_head {
                for d in 0..head_dim {
                    q_3d[h * seq_len * head_dim + s * head_dim + d] =
                        q_node.data[s * n_head * head_dim + h * head_dim + d];
                }
            }
        }

        // Reshape K to [n_kv_head, seq_len, head_dim], then expand to [n_head, seq_len, head_dim]
        let mut k_3d = vec![0.0; seq_len * n_kv_head * head_dim];
        for s in 0..seq_len {
            for h in 0..n_kv_head {
                for d in 0..head_dim {
                    k_3d[h * seq_len * head_dim + s * head_dim + d] =
                        k_node.data[s * n_kv_head * head_dim + h * head_dim + d];
                }
            }
        }

        // Expand KV: repeat_interleave dim=0 by `repeat`
        let mut k_expanded = vec![0.0; seq_len * n_head * head_dim];
        for h in 0..n_head {
            let src_h = h / repeat;
            let dst_offset = h * seq_len * head_dim;
            let src_offset = src_h * seq_len * head_dim;
            k_expanded[dst_offset..dst_offset + seq_len * head_dim]
                .copy_from_slice(&k_3d[src_offset..src_offset + seq_len * head_dim]);
        }

        // Same for V
        let mut v_3d = vec![0.0; seq_len * n_kv_head * head_dim];
        for s in 0..seq_len {
            for h in 0..n_kv_head {
                for d in 0..head_dim {
                    v_3d[h * seq_len * head_dim + s * head_dim + d] =
                        v_node.data[s * n_kv_head * head_dim + h * head_dim + d];
                }
            }
        }

        let mut v_expanded = vec![0.0; seq_len * n_head * head_dim];
        for h in 0..n_head {
            let src_h = h / repeat;
            let dst_offset = h * seq_len * head_dim;
            let src_offset = src_h * seq_len * head_dim;
            v_expanded[dst_offset..dst_offset + seq_len * head_dim]
                .copy_from_slice(&v_3d[src_offset..src_offset + seq_len * head_dim]);
        }

        // Scores: Q @ K^T . K^T: [n_head, head_dim, seq_len]
        // scores[h, s1, s2] = sum_d Q[h, s1, d] * K[h, s2, d] * inv_sqrt_d
        let mut scores = vec![0.0; n_head * seq_len * seq_len];
        for h in 0..n_head {
            for s1 in 0..seq_len {
                for s2 in 0..seq_len {
                    let mut sum = 0.0;
                    let q_off = h * seq_len * head_dim + s1 * head_dim;
                    let k_off = h * seq_len * head_dim + s2 * head_dim;
                    for d in 0..head_dim {
                        sum += q_3d[q_off + d] * k_expanded[k_off + d];
                    }
                    scores[h * seq_len * seq_len + s1 * seq_len + s2] = sum * inv_sqrt_d;
                }
            }
        }

        // Add causal mask + softmax
        let mut attn = vec![0.0; n_head * seq_len * seq_len];
        for h in 0..n_head {
            for s1 in 0..seq_len {
                let base = h * seq_len * seq_len + s1 * seq_len;
                // Add mask: for s2 > s1, the mask value is -inf (use -1e9)
                let mut max_val = f32::NEG_INFINITY;
                for s2 in 0..seq_len {
                    let masked = if s2 > s1 {
                        scores[base + s2] - 1e9
                    } else {
                        scores[base + s2] + mask_node.data[s1 * seq_len + s2]
                    };
                    attn[base + s2] = masked;
                    if masked > max_val {
                        max_val = masked;
                    }
                }
                // Softmax
                let mut sum_exp = 0.0;
                for s2 in 0..seq_len {
                    let e = (attn[base + s2] - max_val).exp();
                    attn[base + s2] = e;
                    sum_exp += e;
                }
                for s2 in 0..seq_len {
                    attn[base + s2] /= sum_exp;
                }
            }
        }

        // Output: attn @ V_expanded → [n_head, seq_len, head_dim]
        let mut out_3d = vec![0.0; seq_len * n_head * head_dim];
        for h in 0..n_head {
            for s1 in 0..seq_len {
                for d in 0..head_dim {
                    let mut sum = 0.0;
                    let attn_base = h * seq_len * seq_len + s1 * seq_len;
                    // V is [n_head, seq_len, head_dim], we need V[h, s2, d]
                    // Wait no: V[h, s2, d] - each s2 has its own V
                    // attn_base + s2 is the weight for V[h, s2, d]
                    for s2 in 0..seq_len {
                        sum += attn[attn_base + s2]
                            * v_expanded[h * seq_len * head_dim + s2 * head_dim + d];
                    }
                    out_3d[h * seq_len * head_dim + s1 * head_dim + d] = sum;
                }
            }
        }

        // Transpose back to [seq_len, n_head, head_dim] → flatten to [seq_len, n_head * head_dim]
        let mut output = vec![0.0; seq_len * n_head * head_dim];
        for s in 0..seq_len {
            for h in 0..n_head {
                for d in 0..head_dim {
                    output[s * n_head * head_dim + h * head_dim + d] =
                        out_3d[h * seq_len * head_dim + s * head_dim + d];
                }
            }
        }

        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            data: std::sync::Arc::new(output),
            grad: vec![0.0; seq_len * n_head * head_dim],
            shape: vec![seq_len, n_head * head_dim],
            op: Op::MultiHeadAttention {
                q: q_id,
                k: k_id,
                v: v_id,
                mask: mask_id,
                n_head,
                n_kv_head,
                head_dim,
            },
            extra: Some(std::sync::Arc::new(attn)),
        });
        id
    }

    /// SelectRow: extrae la fila `row_idx` de un tensor 2D [N, D] → [1, D].
    pub fn select_row(&mut self, x_id: NodeId, row_idx: usize) -> NodeId {
        let x_node = &self.nodes[x_id.0];
        assert_eq!(x_node.shape.len(), 2, "SelectRow requiere shape 2D");
        assert!(
            row_idx < x_node.shape[0],
            "SelectRow: row_idx fuera de rango"
        );
        let d = x_node.shape[1];
        let data = x_node.data[row_idx * d..(row_idx + 1) * d].to_vec();
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            data: std::sync::Arc::new(data),
            grad: vec![0.0; d],
            shape: vec![1, d],
            op: Op::SelectRow(x_id, row_idx),
            extra: None,
        });
        id
    }

    /// VICReg: Variance-Invariance-Covariance Regularization (TRAIN-03).
    /// z: [seq_len, hidden], coeff: peso de la regularización.
    /// Output: scalar loss = coeff * (var_loss + cov_loss).
    pub fn vicreg(&mut self, z_id: NodeId, coeff: f32) -> NodeId {
        let z_node = &self.nodes[z_id.0];
        let seq_len = z_node.shape[0];
        let dim = z_node.shape[1];
        let eps = 1.0;

        // Mean across sequence: mean[d] = 1/seq_len * sum_s z[s, d]
        let mut mean = vec![0.0; dim];
        for s in 0..seq_len {
            for d in 0..dim {
                mean[d] += z_node.data[s * dim + d];
            }
        }
        for d in 0..dim {
            mean[d] /= seq_len as f32;
        }

        // Centered data: c[s, d] = z[s, d] - mean[d]
        let mut centered = vec![0.0; seq_len * dim];
        for s in 0..seq_len {
            for d in 0..dim {
                centered[s * dim + d] = z_node.data[s * dim + d] - mean[d];
            }
        }

        // Variance: var[d] = 1/seq_len * sum_s centered[s, d]^2
        let mut var = vec![0.0; dim];
        for s in 0..seq_len {
            for d in 0..dim {
                var[d] += centered[s * dim + d] * centered[s * dim + d];
            }
        }
        for d in 0..dim {
            var[d] /= seq_len as f32;
        }

        // Variance loss: 1/D * sum_d max(0, 1 - sqrt(var[d] + eps))
        let mut var_loss = 0.0;
        let mut sqrt_var = vec![0.0; dim];
        for d in 0..dim {
            sqrt_var[d] = (var[d] + eps).sqrt();
            if sqrt_var[d] < 1.0 {
                var_loss += 1.0 - sqrt_var[d];
            }
        }
        var_loss /= dim as f32;

        // Covariance: C[i][j] = 1/seq_len * sum_s centered[s, i] * centered[s, j]
        // Only off-diagonal contributes: cov_loss = 1/D * sum_{i != j} C[i][j]^2
        let mut cov_loss = 0.0;
        let mut cov = vec![vec![0.0; dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = 0.0;
                for s in 0..seq_len {
                    sum += centered[s * dim + i] * centered[s * dim + j];
                }
                cov[i][j] = sum / seq_len as f32;
                if i != j {
                    cov_loss += cov[i][j] * cov[i][j];
                }
            }
        }
        cov_loss /= dim as f32;

        let loss = coeff * (var_loss + cov_loss);

        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            data: std::sync::Arc::new(vec![loss]),
            grad: vec![0.0],
            shape: vec![1],
            op: Op::VICReg(z_id, coeff),
            extra: None,
        });
        id
    }

    /// KL Divergence: KL(student || teacher) con temperatura
    /// student, teacher: [num_classes], temp: f32
    /// forward: KL(softmax(s/temp) || softmax(t/temp))
    pub fn kl_div(&mut self, student_id: NodeId, teacher_id: NodeId, temperature: f32) -> NodeId {
        let s_node = &self.nodes[student_id.0];
        let t_node = &self.nodes[teacher_id.0];
        assert_eq!(
            s_node.data.len(),
            t_node.data.len(),
            "Student y teacher deben tener mismas dimensiones"
        );
        let n = s_node.data.len();

        // Softmax con temperatura para student
        let s_max = s_node
            .data
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut s_exp = vec![0.0; n];
        let mut s_sum = 0.0;
        for i in 0..n {
            let e = ((s_node.data[i] - s_max) / temperature).exp();
            s_exp[i] = e;
            s_sum += e;
        }

        // Softmax con temperatura para teacher (teacher se considera constante, sin gradiente)
        let t_max = t_node
            .data
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut t_exp = vec![0.0; n];
        let mut t_sum = 0.0;
        for i in 0..n {
            let e = ((t_node.data[i] - t_max) / temperature).exp();
            t_exp[i] = e;
            t_sum += e;
        }

        // KL(P_teacher || P_student) donde teacher es la distribución objetivo
        let mut loss = 0.0;
        for i in 0..n {
            let p_t = t_exp[i] / t_sum;
            let p_s = s_exp[i] / s_sum;
            if p_t > 1e-10 && p_s > 1e-10 {
                loss += p_t * (p_t / p_s).ln();
            }
        }

        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            data: std::sync::Arc::new(vec![loss]),
            grad: vec![0.0],
            shape: vec![1],
            op: Op::KLDiv(student_id, teacher_id, temperature),
            extra: None,
        });
        id
    }

    /// KLDivDirect: KL(p||q) sin softmax. p,q son distribuciones de probabilidad [N].
    pub fn kl_div_direct(&mut self, p_id: NodeId, q_id: NodeId) -> NodeId {
        let p_node = &self.nodes[p_id.0];
        let q_node = &self.nodes[q_id.0];
        assert_eq!(p_node.data.len(), q_node.data.len());
        let n = p_node.data.len();

        let mut loss = 0.0;
        for i in 0..n {
            let pi = p_node.data[i].max(1e-10);
            let qi = q_node.data[i].max(1e-10);
            loss += pi * (pi / qi).ln();
        }

        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            data: std::sync::Arc::new(vec![loss]),
            grad: vec![0.0],
            shape: vec![1],
            op: Op::KLDivDirect(p_id, q_id),
            extra: None,
        });
        id
    }

    /// Pérdida de Entropía Cruzada sobre un vector de Logits 1D.
    pub fn cross_entropy(&mut self, logits_id: NodeId, target: usize) -> NodeId {
        let logits_node = &self.nodes[logits_id.0];
        assert!(logits_node.data.len() > target, "Target fuera de rango");

        let max_logit = logits_node
            .data
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0;
        let mut exps = vec![0.0; logits_node.data.len()];
        for i in 0..logits_node.data.len() {
            let e = (logits_node.data[i] - max_logit).exp();
            exps[i] = e;
            sum_exp += e;
        }

        let target_prob = exps[target] / sum_exp;
        let loss = -target_prob.max(1e-7).ln();

        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            data: std::sync::Arc::new(vec![loss]),
            grad: vec![0.0],
            shape: vec![1],
            op: Op::CrossEntropy(logits_id, target),
            extra: None,
        });
        id
    }

    pub fn backward(&mut self, root: NodeId) {
        // Inicializa el gradiente del nodo raíz (usualmente la pérdida de forma [1])
        for g in &mut self.nodes[root.0].grad {
            *g = 1.0;
        }

        for i in (0..=root.0).rev() {
            if i & 15 == 0 && self.should_abort() {
                return;
            }
            let op = self.nodes[i].op.clone();
            if matches!(op, Op::Leaf) {
                continue;
            }

            match op {
                Op::Add(lhs, rhs) => {
                    // dX = dZ, dY = dZ (elemento por elemento con AXPY)
                    let (z, left, right) = self.get_three_mut(NodeId(i), lhs, rhs);
                    unsafe {
                        avx_math::axpy_avx2(&mut left.grad, 1.0, &z.grad);
                        avx_math::axpy_avx2(&mut right.grad, 1.0, &z.grad);
                    }
                }
                Op::Mul(lhs, rhs) => {
                    // dX = dZ * Y, dY = dZ * X
                    let (z, left, right) = self.get_three_mut(NodeId(i), lhs, rhs);
                    for j in 0..z.grad.len() {
                        let dz = z.grad[j];
                        if dz != 0.0 {
                            left.grad[j] += dz * right.data[j];
                            right.grad[j] += dz * left.data[j];
                        }
                    }
                }
                Op::MatMul(x_id, w_id) => {
                    // Z = X * W^T
                    // dX = dZ * W   donde dZ: [M, N], W: [N, K] -> dX: [M, K]
                    // dW = dZ^T * X donde dZ: [M, N], X: [M, K] -> dW: [N, K]
                    let z = self.nodes[i].clone();

                    let (x_node, w_node) = self.get_two_mut(x_id, w_id);
                    let m = z.shape[0];
                    let n = z.shape[1];
                    let k = x_node.shape[1];

                    // 1. dX = dZ * W
                    x_node
                        .grad
                        .chunks_mut(k)
                        .enumerate()
                        .for_each(|(i_m, x_grad_row)| {
                            for j_n in 0..n {
                                let dz_val = z.grad[i_m * n + j_n];
                                if dz_val != 0.0 {
                                    let w_data_row = &w_node.data[j_n * k..(j_n + 1) * k];
                                    unsafe {
                                        avx_math::axpy_avx2(x_grad_row, dz_val, w_data_row);
                                    }
                                }
                            }
                        });

                    // 2. dW = dZ^T * X
                    w_node
                        .grad
                        .chunks_mut(k)
                        .enumerate()
                        .for_each(|(j_n, w_grad_row)| {
                            for i_m in 0..m {
                                let dz_val = z.grad[i_m * n + j_n];
                                if dz_val != 0.0 {
                                    let x_data_row = &x_node.data[i_m * k..(i_m + 1) * k];
                                    unsafe {
                                        avx_math::axpy_avx2(w_grad_row, dz_val, x_data_row);
                                    }
                                }
                            }
                        });
                }
                Op::SiLU(x_id) => {
                    let z = self.nodes[i].clone();
                    let x_node = &mut self.nodes[x_id.0];
                    for j in 0..z.data.len() {
                        let dz = z.grad[j];
                        if dz == 0.0 {
                            continue;
                        }
                        let x = x_node.data[j];
                        let sig = 1.0 / (1.0 + (-x).exp());
                        // d(x * sig)/dx = sig + x * sig * (1 - sig)
                        let grad_x = sig * (1.0 + x * (1.0 - sig));
                        x_node.grad[j] += dz * grad_x;
                    }
                }
                Op::CrossEntropy(logits_id, target) => {
                    let z = self.nodes[i].clone();
                    let logits_node = &mut self.nodes[logits_id.0];

                    let max_logit = logits_node
                        .data
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0;
                    let mut exps = vec![0.0; logits_node.data.len()];
                    for j in 0..logits_node.data.len() {
                        let e = (logits_node.data[j] - max_logit).exp();
                        exps[j] = e;
                        sum_exp += e;
                    }

                    let dz = z.grad[0]; // Escalar loss gradiente
                    for j in 0..logits_node.data.len() {
                        let prob = exps[j] / sum_exp;
                        let indicator = if j == target { 1.0 } else { 0.0 };
                        // dLoss/dLogit_j = Prob_j - Y_j
                        logits_node.grad[j] += dz * (prob - indicator);
                    }
                }
                Op::STEQuantize(x_id, _scale_id) => {
                    // STE: dL/dx = dL/dz (el gradiente pasa como si round/clamp fuera identidad)
                    let z = self.nodes[i].clone();
                    let x_node = &mut self.nodes[x_id.0];
                    unsafe {
                        avx_math::axpy_avx2(&mut x_node.grad, 1.0, &z.grad);
                    }
                }
                Op::RMSNorm(x_id, w_id, eps) => {
                    let z = self.nodes[i].clone();
                    let (x_node, w_node) = self.get_two_mut(x_id, w_id);
                    let n = w_node.data.len();
                    let total_elements = z.data.len();
                    let m = total_elements / n;
                    let inv_n = 1.0 / n as f32;

                    for i in 0..m {
                        let row_start = i * n;
                        let mut sum_sq: f32 = 0.0;
                        for k in 0..n {
                            let v = x_node.data[row_start + k];
                            sum_sq += v * v;
                        }
                        let rms_scale = 1.0 / ((sum_sq / n as f32) + eps).sqrt();

                        let mut sum_dz_w_x = 0.0;
                        for k in 0..n {
                            let dz_val = z.grad[row_start + k];
                            sum_dz_w_x += dz_val * w_node.data[k] * x_node.data[row_start + k];
                        }
                        let scale_correction = rms_scale.powi(3) * sum_dz_w_x * inv_n;

                        for j in 0..n {
                            let idx = row_start + j;
                            let dz_val = z.grad[idx];
                            x_node.grad[idx] += dz_val * w_node.data[j] * rms_scale
                                - w_node.data[j] * x_node.data[idx] * scale_correction;

                            w_node.grad[j] += dz_val * x_node.data[idx] * rms_scale;
                        }
                    }
                }
                Op::KLDiv(student_id, teacher_id, temperature) => {
                    let z = self.nodes[i].clone();
                    let temp = temperature;

                    // Clonar teacher y student data para evitar el borrow checker
                    let s_data: Vec<f32>;
                    let t_data: Vec<f32>;
                    {
                        let s_node_ref = &self.nodes[student_id.0];
                        s_data = s_node_ref.data.as_ref().clone();
                    }
                    {
                        let t_node = &self.nodes[teacher_id.0];
                        t_data = t_node.data.as_ref().clone();
                    }

                    let n = s_data.len();
                    let s_node = &mut self.nodes[student_id.0];

                    // Softmax con temperatura para student (usando datos clonados)
                    let s_max = s_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut s_exp = vec![0.0; n];
                    let mut s_sum = 0.0;
                    for j in 0..n {
                        let e = ((s_data[j] - s_max) / temp).exp();
                        s_exp[j] = e;
                        s_sum += e;
                    }

                    // Softmax con temperatura para teacher
                    let t_max = t_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut t_exp = vec![0.0; n];
                    let mut t_sum = 0.0;
                    for j in 0..n {
                        let e = ((t_data[j] - t_max) / temp).exp();
                        t_exp[j] = e;
                        t_sum += e;
                    }

                    let dz = z.grad[0];
                    // dKL/ds_j = (p_s_j - p_t_j) / temp  (derivada estándar de KL con softmax)
                    for j in 0..n {
                        let p_s = s_exp[j] / s_sum;
                        let p_t = t_exp[j] / t_sum;
                        s_node.grad[j] += dz * (p_s - p_t) / temp;
                    }
                }
                Op::KLDivDirect(p_id, q_id) => {
                    let z = self.nodes[i].clone();
                    let p_data: Vec<f32>;
                    let q_data: Vec<f32>;
                    {
                        let pn = &self.nodes[p_id.0];
                        let qn = &self.nodes[q_id.0];
                        p_data = pn.data.as_ref().clone();
                        q_data = qn.data.as_ref().clone();
                    }
                    let dz = z.grad[0];
                    let n = p_data.len();
                    let p_node = &mut self.nodes[p_id.0];
                    for j in 0..n {
                        let pi = p_data[j].max(1e-10);
                        let qi = q_data[j].max(1e-10);
                        p_node.grad[j] += dz * ((pi / qi).ln() + 1.0);
                    }
                }
                Op::Softmax(x_id) => {
                    // dL/dx_i = softmax_i * (dL/ds_i - sum_j dL/ds_j * softmax_j)
                    let z = self.nodes[i].clone();
                    let x_node = &mut self.nodes[x_id.0];
                    let inner = if z.shape.len() >= 2 {
                        z.shape[1]
                    } else {
                        z.shape[0]
                    };
                    let outer = if z.shape.len() >= 2 { z.shape[0] } else { 1 };

                    for batch in 0..outer {
                        let start = batch * inner;
                        let end = start + inner;
                        // Compute sum_j dL/ds_j * softmax_j
                        let mut sum_ds_p = 0.0;
                        for j in start..end {
                            sum_ds_p += z.grad[j] * z.data[j];
                        }
                        for j in start..end {
                            x_node.grad[j] += z.data[j] * (z.grad[j] - sum_ds_p);
                        }
                    }
                }
                Op::Reshape(x_id, _) => {
                    // dL/dIn = dL/dOut (reshape es solo una vista, los gradientes son 1:1)
                    let z = self.nodes[i].clone();
                    let x_node = &mut self.nodes[x_id.0];
                    unsafe {
                        avx_math::axpy_avx2(&mut x_node.grad, 1.0, &z.grad);
                    }
                }
                Op::Transpose(x_id) => {
                    // dL/dIn_ij = dL/dOut_ji (transpose grad back)
                    let z = self.nodes[i].clone();
                    let x_node = &mut self.nodes[x_id.0];
                    let m = z.shape[1]; // original M
                    let n = z.shape[0]; // original N
                    for i in 0..m {
                        for j in 0..n {
                            x_node.grad[i * n + j] += z.grad[j * m + i];
                        }
                    }
                }
                Op::MultiHeadAttention {
                    q,
                    k,
                    v,
                    mask: _,
                    n_head,
                    n_kv_head,
                    head_dim,
                } => {
                    let z = self.nodes[i].clone();
                    let seq_len = z.shape[0];
                    let repeat = n_head / n_kv_head;

                    // 1. Obtener datos guardados: attn weights y Q/K/V data se recomputan parcialmente
                    // Necesitamos Q_data, K_data, V_data y attn para backward
                    let q_data: Vec<f32>;
                    let k_data: Vec<f32>;
                    let v_data: Vec<f32>;
                    {
                        let qn = &self.nodes[q.0];
                        let kn = &self.nodes[k.0];
                        let vn = &self.nodes[v.0];
                        q_data = qn.data.as_ref().clone();
                        k_data = kn.data.as_ref().clone();
                        v_data = vn.data.as_ref().clone();
                    }

                    let (q2, k2, v2) = self.get_three_mut(q, k, v);

                    // Reconstruir Q_3d [n_head, seq_len, head_dim]
                    let mut q_3d = vec![0.0; n_head * seq_len * head_dim];
                    for s in 0..seq_len {
                        for h in 0..n_head {
                            for d in 0..head_dim {
                                q_3d[h * seq_len * head_dim + s * head_dim + d] =
                                    q_data[s * n_head * head_dim + h * head_dim + d];
                            }
                        }
                    }

                    // Reconstruir K_3d expandida y V_3d expandida
                    let mut k_3d_base = vec![0.0; seq_len * n_kv_head * head_dim];
                    let mut v_3d_base = vec![0.0; seq_len * n_kv_head * head_dim];
                    for s in 0..seq_len {
                        for h in 0..n_kv_head {
                            for d in 0..head_dim {
                                k_3d_base[h * seq_len * head_dim + s * head_dim + d] =
                                    k_data[s * n_kv_head * head_dim + h * head_dim + d];
                                v_3d_base[h * seq_len * head_dim + s * head_dim + d] =
                                    v_data[s * n_kv_head * head_dim + h * head_dim + d];
                            }
                        }
                    }

                    // Expandir
                    let mut k_exp = vec![0.0; n_head * seq_len * head_dim];
                    let mut v_exp = vec![0.0; n_head * seq_len * head_dim];
                    for h in 0..n_head {
                        let src = h / repeat;
                        let dst_off = h * seq_len * head_dim;
                        let src_off = src * seq_len * head_dim;
                        k_exp[dst_off..dst_off + seq_len * head_dim]
                            .copy_from_slice(&k_3d_base[src_off..src_off + seq_len * head_dim]);
                        v_exp[dst_off..dst_off + seq_len * head_dim]
                            .copy_from_slice(&v_3d_base[src_off..src_off + seq_len * head_dim]);
                    }

                    // Reconstruir attn weights (necesario para softmax backward)
                    let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();
                    let mut attn = vec![0.0; n_head * seq_len * seq_len];
                    for h in 0..n_head {
                        for s1 in 0..seq_len {
                            let base = h * seq_len * seq_len + s1 * seq_len;
                            // Compute scores
                            let mut scores = vec![0.0; seq_len];
                            for s2 in 0..seq_len {
                                let mut sum = 0.0;
                                let q_off = h * seq_len * head_dim + s1 * head_dim;
                                let k_off = h * seq_len * head_dim + s2 * head_dim;
                                for d in 0..head_dim {
                                    sum += q_3d[q_off + d] * k_exp[k_off + d];
                                }
                                scores[s2] = sum * inv_sqrt_d;
                            }
                            // Causal mask + softmax
                            let mut max_val = f32::NEG_INFINITY;
                            for s2 in 0..seq_len {
                                let masked = if s2 > s1 {
                                    scores[s2] - 1e9
                                } else {
                                    scores[s2]
                                };
                                attn[base + s2] = masked;
                                if masked > max_val {
                                    max_val = masked;
                                }
                            }
                            let mut sum_exp = 0.0;
                            for s2 in 0..seq_len {
                                let e = (attn[base + s2] - max_val).exp();
                                attn[base + s2] = e;
                                sum_exp += e;
                            }
                            for s2 in 0..seq_len {
                                attn[base + s2] /= sum_exp;
                            }
                        }
                    }

                    // d_out_3d: [S, H, D] → [H, S, D]
                    let mut d_out_3d = vec![0.0; n_head * seq_len * head_dim];
                    for s in 0..seq_len {
                        for h in 0..n_head {
                            for d in 0..head_dim {
                                d_out_3d[h * seq_len * head_dim + s * head_dim + d] =
                                    z.grad[s * n_head * head_dim + h * head_dim + d];
                            }
                        }
                    }

                    // d_attn[h, s1, s2] = sum_d d_out[h, s1, d] * V_exp[h, s2, d]
                    // = (d_out @ V^T) for each h
                    let mut d_attn = vec![0.0; n_head * seq_len * seq_len];
                    for h in 0..n_head {
                        for s1 in 0..seq_len {
                            for s2 in 0..seq_len {
                                let mut sum = 0.0;
                                let do_off = h * seq_len * head_dim + s1 * head_dim;
                                let v_off = h * seq_len * head_dim + s2 * head_dim;
                                for d in 0..head_dim {
                                    sum += d_out_3d[do_off + d] * v_exp[v_off + d];
                                }
                                d_attn[h * seq_len * seq_len + s1 * seq_len + s2] = sum;
                            }
                        }
                    }

                    // d_scores = softmax_backward(d_attn, attn)
                    // d_scores[h, s1, s2] = attn[s2] * (d_attn[s2] - sum_k attn[k] * d_attn[k])
                    let mut d_scores = vec![0.0; n_head * seq_len * seq_len];
                    for h in 0..n_head {
                        for s1 in 0..seq_len {
                            let base = h * seq_len * seq_len + s1 * seq_len;
                            let mut sum_attn_dattn = 0.0;
                            for s2 in 0..seq_len {
                                sum_attn_dattn += attn[base + s2] * d_attn[base + s2];
                            }
                            for s2 in 0..seq_len {
                                d_scores[base + s2] =
                                    attn[base + s2] * (d_attn[base + s2] - sum_attn_dattn);
                            }
                        }
                    }

                    // d_Q[h, s1, d] = sum_{s2} d_scores[h, s1, s2] * K_exp[h, s2, d] / sqrt(D)
                    let mut d_q_3d = vec![0.0; n_head * seq_len * head_dim];
                    for h in 0..n_head {
                        for s1 in 0..seq_len {
                            for d in 0..head_dim {
                                let mut sum = 0.0;
                                let base = h * seq_len * seq_len + s1 * seq_len;
                                for s2 in 0..seq_len {
                                    sum += d_scores[base + s2]
                                        * k_exp[h * seq_len * head_dim + s2 * head_dim + d];
                                }
                                d_q_3d[h * seq_len * head_dim + s1 * head_dim + d] =
                                    sum * inv_sqrt_d;
                            }
                        }
                    }

                    // d_K_exp[h, s2, d] = sum_{s1} d_scores[h, s1, s2] * Q[h, s1, d] / sqrt(D)
                    let mut d_k_exp = vec![0.0; n_head * seq_len * head_dim];
                    for h in 0..n_head {
                        for s2 in 0..seq_len {
                            for d in 0..head_dim {
                                let mut sum = 0.0;
                                for s1 in 0..seq_len {
                                    sum += d_scores[h * seq_len * seq_len + s1 * seq_len + s2]
                                        * q_3d[h * seq_len * head_dim + s1 * head_dim + d];
                                }
                                d_k_exp[h * seq_len * head_dim + s2 * head_dim + d] =
                                    sum * inv_sqrt_d;
                            }
                        }
                    }

                    // d_V_exp[h, s2, d] = sum_{s1} attn[h, s1, s2] * d_out[h, s1, d]
                    let mut d_v_exp = vec![0.0; n_head * seq_len * head_dim];
                    for h in 0..n_head {
                        for s2 in 0..seq_len {
                            for d in 0..head_dim {
                                let mut sum = 0.0;
                                for s1 in 0..seq_len {
                                    sum += attn[h * seq_len * seq_len + s1 * seq_len + s2]
                                        * d_out_3d[h * seq_len * head_dim + s1 * head_dim + d];
                                }
                                d_v_exp[h * seq_len * head_dim + s2 * head_dim + d] = sum;
                            }
                        }
                    }

                    // Reduce K/V gradients: sum over the repeat group
                    let mut d_k_3d = vec![0.0; n_kv_head * seq_len * head_dim];
                    let mut d_v_3d = vec![0.0; n_kv_head * seq_len * head_dim];
                    for src_h in 0..n_kv_head {
                        for s in 0..seq_len {
                            for d in 0..head_dim {
                                let mut k_sum = 0.0;
                                let mut v_sum = 0.0;
                                for ri in 0..repeat {
                                    let h = src_h * repeat + ri;
                                    k_sum += d_k_exp[h * seq_len * head_dim + s * head_dim + d];
                                    v_sum += d_v_exp[h * seq_len * head_dim + s * head_dim + d];
                                }
                                d_k_3d[src_h * seq_len * head_dim + s * head_dim + d] = k_sum;
                                d_v_3d[src_h * seq_len * head_dim + s * head_dim + d] = v_sum;
                            }
                        }
                    }

                    // Transpose d_Q back to [seq_len, n_head, head_dim] → flatten
                    for s in 0..seq_len {
                        for h in 0..n_head {
                            for d in 0..head_dim {
                                q2.grad[s * n_head * head_dim + h * head_dim + d] +=
                                    d_q_3d[h * seq_len * head_dim + s * head_dim + d];
                            }
                        }
                    }

                    // d_K: [n_kv_head, seq_len, head_dim] → [seq_len, n_kv_head, head_dim]
                    for s in 0..seq_len {
                        for h in 0..n_kv_head {
                            for d in 0..head_dim {
                                k2.grad[s * n_kv_head * head_dim + h * head_dim + d] +=
                                    d_k_3d[h * seq_len * head_dim + s * head_dim + d];
                            }
                        }
                    }

                    // Same for V
                    for s in 0..seq_len {
                        for h in 0..n_kv_head {
                            for d in 0..head_dim {
                                v2.grad[s * n_kv_head * head_dim + h * head_dim + d] +=
                                    d_v_3d[h * seq_len * head_dim + s * head_dim + d];
                            }
                        }
                    }
                }
                Op::VICReg(z_id, coeff) => {
                    let z_node = &self.nodes[z_id.0];
                    let seq_len = z_node.shape[0];
                    let dim = z_node.shape[1];
                    let n_f = seq_len as f32;
                    let d_f = dim as f32;
                    let dz = self.nodes[i].grad[0];

                    let mut mean = vec![0.0; dim];
                    for s in 0..seq_len {
                        for d in 0..dim {
                            mean[d] += z_node.data[s * dim + d];
                        }
                    }
                    for d in 0..dim {
                        mean[d] /= n_f;
                    }

                    let mut centered = vec![0.0; seq_len * dim];
                    for s in 0..seq_len {
                        for d in 0..dim {
                            centered[s * dim + d] = z_node.data[s * dim + d] - mean[d];
                        }
                    }

                    let mut var = vec![0.0; dim];
                    for s in 0..seq_len {
                        for d in 0..dim {
                            var[d] += centered[s * dim + d] * centered[s * dim + d];
                        }
                    }
                    for d in 0..dim {
                        var[d] /= n_f;
                    }

                    let mut cov = vec![vec![0.0; dim]; dim];
                    for i in 0..dim {
                        for j in 0..dim {
                            let mut sum = 0.0;
                            for s in 0..seq_len {
                                sum += centered[s * dim + i] * centered[s * dim + j];
                            }
                            cov[i][j] = sum / n_f;
                        }
                    }

                    for s in 0..seq_len {
                        for d_idx in 0..dim {
                            let sv = (var[d_idx] + 1.0).sqrt();
                            let mut g = 0.0;
                            if sv < 1.0 {
                                g -= centered[s * dim + d_idx] / (n_f * sv * d_f);
                            }
                            let mut cov_grad = 0.0;
                            for j in 0..dim {
                                if j != d_idx {
                                    cov_grad += cov[d_idx][j] * centered[s * dim + j];
                                }
                            }
                            g += (4.0 / (d_f * n_f)) * cov_grad;
                            self.nodes[z_id.0].grad[s * dim + d_idx] += dz * coeff * g;
                        }
                    }
                }
                Op::SelectRow(x_id, row_idx) => {
                    let z = self.nodes[i].clone();
                    let x_node = &mut self.nodes[x_id.0];
                    let d = z.shape[1];
                    let start = row_idx * d;
                    for j in 0..d {
                        x_node.grad[start + j] += z.grad[j];
                    }
                }
                Op::TernaryLinear {
                    x: x_id,
                    ref packed_w,
                    ref scales,
                    n_in,
                    n_out,
                } => {
                    let z = self.nodes[i].clone();
                    let x_node = &mut self.nodes[x_id.0];
                    let m = z.shape[0];
                    let blocks_per_row = n_in / 16 + if n_in % 16 != 0 { 1 } else { 0 };
                    // dx[row][col] += sum_j dy[row][j] * w_fp32[j][col]
                    // w_fp32[j][col] = ternary(packed_w[j][col]) * scales[j]
                    for row in 0..m {
                        for j in 0..n_out {
                            let dy_val = z.grad[row * n_out + j] * scales[j];
                            if dy_val == 0.0 {
                                continue;
                            }
                            let w_off = j * blocks_per_row;
                            for b in 0..blocks_per_row {
                                let block = packed_w[w_off + b];
                                let base = b * 16;
                                let limit = (n_in - base).min(16);
                                for bit in 0..limit {
                                    let bits = (block >> (bit * 2)) & 3;
                                    let w_val: f32 = match bits {
                                        1 => 1.0,
                                        2 => -1.0,
                                        _ => 0.0,
                                    };
                                    x_node.grad[row * n_in + base + bit] += dy_val * w_val;
                                }
                            }
                        }
                    }
                }
                Op::Leaf => {}
            }
        }
    }

    /// Obtiene los pesos de atención de un nodo MultiHeadAttention, si están disponibles.
    pub fn get_attn_weights(&self, node_id: NodeId) -> Option<&[f32]> {
        if matches!(self.nodes[node_id.0].op, Op::MultiHeadAttention { .. }) {
            self.nodes[node_id.0].extra.as_ref().map(|v| v.as_slice())
        } else {
            None
        }
    }

    /// 🛡️ SANITIZATION: Ensures all gradients are finite and clamped to a safe range.
    /// Mandated by GEMINI.md to prevent "Zero-Sigma" matrix collapse.
    pub fn sanitize_gradients(&mut self, max_grad: f32) {
        for node in &mut self.nodes {
            for g in &mut node.grad {
                if !g.is_finite() {
                    *g = 0.0;
                } else {
                    *g = g.clamp(-max_grad, max_grad);
                }
            }
        }
    }

    /// Checks if all gradients in the tape are finite.
    pub fn is_finite(&self) -> bool {
        for node in &self.nodes {
            for g in &node.grad {
                if !g.is_finite() {
                    return false;
                }
            }
        }
        true
    }

    // Helper functions for safe mutable aliasing split
    fn get_two(&self, id1: NodeId, id2: NodeId) -> (&Node, &Node) {
        (&self.nodes[id1.0], &self.nodes[id2.0])
    }

    // AG-01: get_two_mut/get_three_mut with bounds check to prevent UB.
    fn get_two_mut(&mut self, id1: NodeId, id2: NodeId) -> (&mut Node, &mut Node) {
        assert!(
            id1.0 < self.nodes.len() && id2.0 < self.nodes.len(),
            "NodeId out of bounds"
        );
        assert!(id1.0 != id2.0);
        let ptr = self.nodes.as_mut_ptr();
        unsafe { (&mut *ptr.add(id1.0), &mut *ptr.add(id2.0)) }
    }

    fn get_three_mut(
        &mut self,
        id1: NodeId,
        id2: NodeId,
        id3: NodeId,
    ) -> (&mut Node, &mut Node, &mut Node) {
        assert!(
            id1.0 < self.nodes.len() && id2.0 < self.nodes.len() && id3.0 < self.nodes.len(),
            "NodeId out of bounds"
        );
        assert!(id1.0 != id2.0 && id1.0 != id3.0 && id2.0 != id3.0);
        let ptr = self.nodes.as_mut_ptr();
        unsafe {
            (
                &mut *ptr.add(id1.0),
                &mut *ptr.add(id2.0),
                &mut *ptr.add(id3.0),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_simd_matmul_gradients() {
        let mut tape = Tape::new();
        // X: [1, 3]
        let x_data = vec![1.0, 2.0, 3.0];
        let x = tape.push_leaf(x_data, vec![1, 3]);

        // W: [2, 3] (Para Linear layer, out_features=2, in_features=3)
        let w_data = vec![0.5, 0.5, 0.5, 1.0, 1.0, 1.0];
        let w = tape.push_leaf(w_data, vec![2, 3]);

        // Z = X * W^T => [1, 2]
        let z = tape.linear(x, w);

        // Z[0] = 1*0.5 + 2*0.5 + 3*0.5 = 3.0
        // Z[1] = 1*1 + 2*1 + 3*1 = 6.0
        assert_eq!(tape.nodes[z.0].data.as_ref(), &vec![3.0, 6.0]);

        tape.backward(z);

        // dZ es 1.0. dW = dZ^T * X
        // dW[0, :] = dZ[0]*X = 1.0 * [1,2,3] = [1,2,3]
        // dW[1, :] = dZ[1]*X = 1.0 * [1,2,3] = [1,2,3]
        assert_eq!(tape.nodes[w.0].grad, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        // dX = dZ * W
        // dX[:] = 1.0 * [0.5, 0.5, 0.5] + 1.0 * [1.0, 1.0, 1.0] = [1.5, 1.5, 1.5]
        assert_eq!(tape.nodes[x.0].grad, vec![1.5, 1.5, 1.5]);
    }

    #[test]
    fn audit_silu_gradients() {
        let mut tape = Tape::new();
        // Prueba con X = 0.0 (esperamos f(0) = 0.0, df/dx = 0.5)
        let x = tape.push_leaf(vec![0.0], vec![1]);
        let z = tape.silu(x);

        assert_eq!(tape.nodes[z.0].data[0], 0.0);
        tape.backward(z);
        // sigmoid(0) = 0.5. SiLU'(0) = 0.5 + 0 * ... = 0.5.
        assert_eq!(tape.nodes[x.0].grad[0], 0.5);
    }

    #[test]
    fn audit_cross_entropy_gradients() {
        let mut tape = Tape::new();
        // Logits: [2.0, 1.0, 0.1]
        // Target: index 0 (el valor 2.0)
        let logits = tape.push_leaf(vec![2.0, 1.0, 0.1], vec![3]);
        let loss = tape.cross_entropy(logits, 0);

        tape.backward(loss);

        // Softmax manual para [2.0, 1.0, 0.1]
        // max = 2.0. e^(0) = 1, e^(-1) = 0.367879, e^(-1.9) = 0.149568
        // sum_exp = 1.517447
        // p_0 = 1 / 1.517447 = 0.65899
        // p_1 = 0.367879 / 1.517447 = 0.24243
        // p_2 = 0.149568 / 1.517447 = 0.09856

        let p0 = tape.nodes[logits.0].grad[0] + 1.0; // porque dL/dLogit_0 = p0 - 1
        assert!((p0 - 0.65899).abs() < 1e-4, "Softmax Target Prob falló");

        let p1 = tape.nodes[logits.0].grad[1]; // dL/dLogit_1 = p1 - 0 = p1
        assert!((p1 - 0.24243).abs() < 1e-4, "Softmax Dist Prob falló");
    }

    #[test]
    fn audit_ste_quantize_forward_backward() {
        let mut tape = Tape::new();
        // x = [0.3, -0.8, 1.2, -0.1, 0.0, 3.0]
        let x = tape.push_leaf(vec![0.3, -0.8, 1.2, -0.1, 0.0, 3.0], vec![6]);
        // scale = 1.0
        let s = tape.push_leaf(vec![1.0], vec![1]);

        // Quantized: round(x/1).clamp(-1,1)*1 = [0, -1, 1, 0, 0, 1]
        let q = tape.ste_quantize(x, s);
        let expected: Vec<f32> = vec![0.0, -1.0, 1.0, 0.0, 0.0, 1.0];
        assert_eq!(
            tape.nodes[q.0].data.as_ref(),
            &expected,
            "STE forward falló"
        );

        // Construir loss = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] usando Add
        // Add trabaja elemento a elemento, así que pairwise sum hasta 1 escalar
        let idx0 = tape.push_leaf(vec![tape.nodes[q.0].data[0]], vec![1]);
        let mut acc = idx0;
        for j in 1..6 {
            let val = tape.nodes[q.0].data[j];
            let elem = tape.push_leaf(vec![val], vec![1]);
            acc = tape.add(acc, elem);
        }
        let loss = acc;
        assert_eq!(tape.nodes[loss.0].data[0], 1.0);

        // El gradiente de loss a q es [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        // porque d(sum(q_i))/dq_i = 1.0 para cada i

        // Necesitamos conectar loss con q explícitamente en el grafo.
        // Enfoque: loss usa los valores de q como leaves (no como nodos),
        // así que loss no está conectado al grafo de q.
        // Reparamos: creamos Add(q, zeros) y luego sumamos elementos individuales.
        tape.reset();

        // Test más limpio: STE con un solo escalar
        let x2 = tape.push_leaf(vec![2.0], vec![1]);
        let s2 = tape.push_leaf(vec![1.0], vec![1]);
        let q2 = tape.ste_quantize(x2, s2);
        // q2 = round(2/1).clamp(-1,1)*1 = 1.0
        assert_eq!(tape.nodes[q2.0].data[0], 1.0);

        // loss = q2 (identidad)
        let w = tape.push_leaf(vec![1.0], vec![1]);
        let loss2 = tape.mul(q2, w);
        // loss2 = 1.0 * 1.0 = 1.0
        assert_eq!(tape.nodes[loss2.0].data[0], 1.0);

        tape.backward(loss2);
        // dLoss2/dq2 = w = 1.0, STE: dLoss2/dx2 = dLoss2/dq2 = 1.0
        assert!(
            (tape.nodes[x2.0].grad[0] - 1.0).abs() < 1e-5,
            "STE backward: grad debe ser 1.0, got {}",
            tape.nodes[x2.0].grad[0]
        );
    }

    #[test]
    fn audit_rms_norm_forward_backward() {
        let mut tape = Tape::new();
        // x = [1.0, 2.0, 3.0]
        let x = tape.push_leaf(vec![1.0, 2.0, 3.0], vec![3]);
        // w = [1.0, 1.0, 1.0] (identity weights)
        let w = tape.push_leaf(vec![1.0, 1.0, 1.0], vec![3]);

        let eps = 1e-6;
        let y = tape.rms_norm(x, w, eps);

        // RMSNorm manual: mean_sq = (1+4+9)/3 = 4.666..., rms_scale = 1/sqrt(4.666...+1e-6) = 0.46291
        // y = [0.46291, 0.92582, 1.38873]
        let y_data = tape.nodes[y.0].data.as_ref();
        let sum_sq: f32 = 1.0 + 4.0 + 9.0;
        let rms_scale = 1.0 / ((sum_sq / 3.0) + eps).sqrt();
        assert!(
            (y_data[0] - 1.0 * rms_scale).abs() < 1e-4,
            "RMSNorm forward x0 falló"
        );
        assert!(
            (y_data[1] - 2.0 * rms_scale).abs() < 1e-4,
            "RMSNorm forward x1 falló"
        );
        assert!(
            (y_data[2] - 3.0 * rms_scale).abs() < 1e-4,
            "RMSNorm forward x2 falló"
        );

        // loss = sum(y)
        let ones = tape.push_leaf(vec![1.0; 3], vec![3]);
        let loss = tape.mul(y, ones);
        tape.backward(loss);

        // Solo verificamos que los gradientes son finitos (correctitud numérica)
        assert!(
            tape.nodes[x.0].grad.iter().all(|g| g.is_finite()),
            "RMSNorm x grad no finito"
        );
        assert!(
            tape.nodes[w.0].grad.iter().all(|g| g.is_finite()),
            "RMSNorm w grad no finito"
        );
    }

    #[test]
    fn audit_softmax_1d_forward() {
        let mut tape = Tape::new();
        let x = tape.push_leaf(vec![2.0, 1.0, 0.1], vec![3]);
        let s = tape.softmax(x);
        // max = 2.0. e^0=1, e^-1=0.3679, e^-1.9=0.1496. sum=1.5174
        // p = [0.6590, 0.2424, 0.0986]
        let p = tape.nodes[s.0].data.as_ref();
        assert!((p[0] - 0.65899).abs() < 1e-4);
        assert!((p[1] - 0.24243).abs() < 1e-4);
        assert!((p[2] - 0.09856).abs() < 1e-4);
        assert!((p[0] + p[1] + p[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn audit_softmax_2d_forward() {
        let mut tape = Tape::new();
        // [2, 3] matrix
        let x = tape.push_leaf(vec![2.0, 1.0, 0.1, 0.0, 5.0, -1.0], vec![2, 3]);
        let s = tape.softmax(x);
        let p = tape.nodes[s.0].data.as_ref();
        // Row 0: same as 1D test
        assert!((p[0] - 0.65899).abs() < 1e-4);
        assert!((p[2] - 0.09856).abs() < 1e-4);
        // Row 1: max=5.0. e^-5=0.0067, e^0=1, e^-6=0.0025. sum=1.0092
        // p = [0.0067, 0.9909, 0.0025]
        assert!((p[3] - 0.00674).abs() < 1e-3);
        // Verify each row sums to 1
        assert!((p[0] + p[1] + p[2] - 1.0).abs() < 1e-5);
        assert!((p[3] + p[4] + p[5] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn audit_softmax_backward() {
        let mut tape = Tape::new();
        // Test 1: simple sum-of-softmax = 1, gradient should be 0
        let x = tape.push_leaf(vec![1.0, 2.0, 3.0], vec![3]);
        let s = tape.softmax(x);
        // loss = s[0] (select first element via mul with selector)
        let selector = tape.push_leaf(vec![1.0, 0.0, 0.0], vec![3]);
        let loss_vec = tape.mul(s, selector);
        // Sum the vector to get a scalar loss
        let accum = tape.push_leaf(vec![1.0; 3], vec![3]);
        let loss = tape.mul(loss_vec, accum);
        // loss = s[0] * 1 + 0 * 1 + 0 * 1 = s[0]
        tape.backward(loss);
        assert!(tape.nodes[x.0].grad.iter().all(|g| g.is_finite()));
    }

    #[test]
    fn audit_transpose_forward_backward() {
        let mut tape = Tape::new();
        // [2, 3] -> [3, 2]
        let x = tape.push_leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let xt = tape.transpose(x);
        let out = tape.nodes[xt.0].data.as_ref();
        assert_eq!(tape.nodes[xt.0].shape, vec![3, 2]);
        // Expected: [[1, 4], [2, 5], [3, 6]]
        assert_eq!(out[0], 1.0); // [0,0]
        assert_eq!(out[1], 4.0); // [0,1]
        assert_eq!(out[2], 2.0); // [1,0]
        assert_eq!(out[3], 5.0); // [1,1]
        assert_eq!(out[4], 3.0); // [2,0]
        assert_eq!(out[5], 6.0); // [2,1]

        // loss = sum of all elements
        let ones = tape.push_leaf(vec![1.0; 6], vec![3, 2]);
        let loss = tape.mul(xt, ones);
        tape.backward(loss);
        // dL/dX = dL/dOut^T = ones^T = all-ones [2, 3]
        for g in &tape.nodes[x.0].grad {
            assert!((g - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn audit_reshape_forward_backward() {
        let mut tape = Tape::new();
        // [2, 3] -> [3, 2] (elementos diferentes)
        let x = tape.push_leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let r = tape.reshape(x, vec![3, 2]);
        assert_eq!(tape.nodes[r.0].shape, vec![3, 2]);
        assert_eq!(tape.nodes[r.0].data[0], 1.0);
        assert_eq!(tape.nodes[r.0].data[5], 6.0);
        // loss = sum
        let ones = tape.push_leaf(vec![1.0; 6], vec![3, 2]);
        let loss = tape.mul(r, ones);
        tape.backward(loss);
        assert!(tape.nodes[x.0].grad.iter().all(|g| (g - 1.0).abs() < 1e-5));
    }

    #[test]
    fn audit_kl_div_forward_backward() {
        let mut tape = Tape::new();
        // student logits
        let s = tape.push_leaf(vec![1.0, 0.0, -1.0], vec![3]);
        // teacher logits (constantes)
        let t = tape.push_leaf(vec![2.0, 0.5, -0.5], vec![3]);
        let temp = 1.0;

        let kl = tape.kl_div(s, t, temp);
        // KL >= 0, debe ser finito
        let kl_val = tape.nodes[kl.0].data[0];
        assert!(
            kl_val.is_finite() && kl_val >= 0.0,
            "KL debe ser >= 0, got {}",
            kl_val
        );

        tape.backward(kl);
        // Gradientes deben ser finitos
        assert!(
            tape.nodes[s.0].grad.iter().all(|g| g.is_finite()),
            "KL student grad no finito"
        );
        // Teacher no debe tener gradiente (es constante en el forward)
        assert_eq!(
            tape.nodes[t.0].grad.iter().all(|&g| g == 0.0),
            true,
            "Teacher no debe tener gradiente"
        );
    }

    #[test]
    fn audit_mha_forward_backward() {
        let mut tape = Tape::new();
        let seq_len = 3;
        let n_head = 2;
        let n_kv_head = 1;
        let head_dim = 4;
        let hd = n_head * head_dim;

        // Q: [3, 8], K/V: [3, 4]
        let q_data: Vec<f32> = (0..seq_len * hd).map(|i| i as f32 * 0.1).collect();
        let k_data: Vec<f32> = (0..seq_len * n_kv_head * head_dim)
            .map(|i| i as f32 * 0.05)
            .collect();
        let v_data: Vec<f32> = (0..seq_len * n_kv_head * head_dim)
            .map(|i| i as f32 * 0.02 + 0.5)
            .collect();
        let mask = vec![0.0; seq_len * seq_len]; // zero mask, but causal masking is built-in

        let q = tape.push_leaf(q_data, vec![seq_len, hd]);
        let k = tape.push_leaf(k_data, vec![seq_len, n_kv_head * head_dim]);
        let v = tape.push_leaf(v_data, vec![seq_len, n_kv_head * head_dim]);
        let m = tape.push_leaf(mask, vec![seq_len, seq_len]);

        let out = tape.mha(q, k, v, m, n_head, n_kv_head, head_dim);
        assert_eq!(tape.nodes[out.0].shape, vec![seq_len, hd]);
        assert!(
            tape.nodes[out.0].data.iter().all(|&v| v.is_finite()),
            "MHA forward debe ser finito"
        );

        // loss = sum of output
        let ones = tape.push_leaf(vec![1.0; seq_len * hd], vec![seq_len, hd]);
        let loss = tape.mul(out, ones);
        tape.backward(loss);

        assert!(
            tape.nodes[q.0].grad.iter().all(|&g| g.is_finite()),
            "MHA Q grad finito"
        );
        assert!(
            tape.nodes[k.0].grad.iter().all(|&g| g.is_finite()),
            "MHA K grad finito"
        );
        assert!(
            tape.nodes[v.0].grad.iter().all(|&g| g.is_finite()),
            "MHA V grad finito"
        );

        assert_eq!(tape.nodes[q.0].grad.len(), seq_len * hd);
        assert_eq!(tape.nodes[k.0].grad.len(), seq_len * n_kv_head * head_dim);
    }

    #[test]
    fn audit_vicreg_forward_backward() {
        let mut tape = Tape::new();
        // Repr: [4, 8] — 4 tokens, 8 dims — con valores que tienen varianza baja
        let seq_len = 4;
        let dim = 8;
        // Casi colapsado: todos los vectores son casi iguales
        let mut data = vec![0.0; seq_len * dim];
        for s in 0..seq_len {
            for d in 0..dim {
                data[s * dim + d] = 1.0 + (s as f32 * 0.01); // muy poca varianza
            }
        }
        let z = tape.push_leaf(data, vec![seq_len, dim]);
        let coeff = 1.0;
        let loss = tape.vicreg(z, coeff);
        let loss_val = tape.nodes[loss.0].data[0];
        // VICReg loss debe ser > 0 porque la varianza es muy baja
        assert!(
            loss_val > 0.0,
            "VICReg loss debe ser positivo para baja varianza"
        );
        // loss debe ser finito
        assert!(loss_val.is_finite(), "VICReg loss finito");

        tape.backward(loss);
        // Gradientes deben ser finitos
        assert!(
            tape.nodes[z.0].grad.iter().all(|&g| g.is_finite()),
            "VICReg grad finito"
        );
        // La forma de los gradientes debe coincidir
        assert_eq!(tape.nodes[z.0].grad.len(), seq_len * dim);

        // Test 2: dim=1 (sin covarianza), varianza > 1 → loss debe ser 0
        tape.reset();
        let single_dim = 1;
        let data2 = vec![-1.5, -0.5, 0.5, 1.5];
        let z2 = tape.push_leaf(data2, vec![seq_len, single_dim]);
        let loss2 = tape.vicreg(z2, coeff);
        let loss2_val = tape.nodes[loss2.0].data[0];
        // var=1.25 → sqrt(var+1)=1.5 > 1 → sin penalty de varianza. Sin covarianza (dim=1). loss=0.
        assert!(
            loss2_val.abs() < 1e-5,
            "VICReg dim=1 var>1 debe tener loss 0, got {}",
            loss2_val
        );
    }

    #[test]
    fn audit_select_row_forward_backward() {
        let mut tape = Tape::new();
        // [3, 4] tensor
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let x = tape.push_leaf(data, vec![3, 4]);

        // Seleccionar fila 1
        let row = tape.select_row(x, 1);
        assert_eq!(tape.nodes[row.0].shape, vec![1, 4]);
        assert_eq!(tape.nodes[row.0].data.as_ref(), &vec![5.0, 6.0, 7.0, 8.0]);

        // loss = sum(row)
        let ones = tape.push_leaf(vec![1.0; 4], vec![1, 4]);
        let loss = tape.mul(row, ones);
        tape.backward(loss);

        // Solo la fila 1 debe tener gradiente = 1.0
        let expected_grad = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(tape.nodes[x.0].grad, expected_grad);
    }
}
