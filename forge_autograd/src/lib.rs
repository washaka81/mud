pub mod avx_math;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeId(pub usize);

#[derive(Clone, Copy, Debug)]
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
}

#[derive(Clone, Debug)]
pub struct Node {
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
    pub shape: Vec<usize>,
    pub op: Op,
}

#[derive(Default, Clone)]
pub struct Tape {
    pub nodes: Vec<Node>,
}

impl Tape {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn zero_grad(&mut self) {
        for node in &mut self.nodes {
            for g in &mut node.grad {
                *g = 0.0;
            }
        }
    }

    /// Empuja un tensor de pesos o entrada (Hoja)
    pub fn push_leaf(&mut self, data: Vec<f32>, shape: Vec<usize>) -> NodeId {
        let len = data.len();
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node { data, grad: vec![0.0; len], shape, op: Op::Leaf });
        id
    }

    /// Suma dos tensores elemento por elemento
    pub fn add(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let (lhs_node, rhs_node) = self.get_two(lhs, rhs);
        assert_eq!(lhs_node.shape, rhs_node.shape, "Las formas deben coincidir para Add");
        let len = lhs_node.data.len();
        let mut data = vec![0.0; len];
        
        unsafe {
            // z = x + y equivalente a z = 1.0 * x + y
            data.copy_from_slice(&rhs_node.data);
            avx_math::axpy_avx2(&mut data, 1.0, &lhs_node.data);
        }

        let id = NodeId(self.nodes.len());
        self.nodes.push(Node { data, grad: vec![0.0; len], shape: lhs_node.shape.clone(), op: Op::Add(lhs, rhs) });
        id
    }

    /// Multiplica dos tensores elemento por elemento
    pub fn mul(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let (lhs_node, rhs_node) = self.get_two(lhs, rhs);
        assert_eq!(lhs_node.shape, rhs_node.shape, "Las formas deben coincidir para Mul");
        let len = lhs_node.data.len();
        let mut data = vec![0.0; len];
        
        for i in 0..len {
            data[i] = lhs_node.data[i] * rhs_node.data[i];
        }

        let id = NodeId(self.nodes.len());
        self.nodes.push(Node { data, grad: vec![0.0; len], shape: lhs_node.shape.clone(), op: Op::Mul(lhs, rhs) });
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

        // Z = X * W^T usando AVX2 dot_product
        unsafe {
            for i in 0..m {
                for j in 0..n {
                    let x_row = &x_node.data[i * k .. (i + 1) * k];
                    let w_row = &w_node.data[j * k .. (j + 1) * k];
                    z_data[i * n + j] = avx_math::dot_product_avx2(x_row, w_row);
                }
            }
        }

        let id = NodeId(self.nodes.len());
        self.nodes.push(Node { 
            data: z_data, 
            grad: vec![0.0; m * n], 
            shape: vec![m, n], 
            op: Op::MatMul(x_id, w_id) 
        });
        id
    }

    /// Activación SiLU: f(x) = x * sigmoid(x)
    pub fn silu(&mut self, x_id: NodeId) -> NodeId {
        let x_node = &self.nodes[x_id.0];
        let mut z_data = vec![0.0; x_node.data.len()];
        
        for i in 0..x_node.data.len() {
            let x = x_node.data[i];
            let sig = 1.0 / (1.0 + (-x).exp());
            z_data[i] = x * sig;
        }

        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            data: z_data,
            grad: vec![0.0; x_node.data.len()],
            shape: x_node.shape.clone(),
            op: Op::SiLU(x_id),
        });
        id
    }

    /// Pérdida de Entropía Cruzada sobre un vector de Logits 1D.
    pub fn cross_entropy(&mut self, logits_id: NodeId, target: usize) -> NodeId {
        let logits_node = &self.nodes[logits_id.0];
        assert!(logits_node.data.len() > target, "Target fuera de rango");

        let max_logit = logits_node.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
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
            data: vec![loss],
            grad: vec![0.0],
            shape: vec![1],
            op: Op::CrossEntropy(logits_id, target),
        });
        id
    }

    pub fn backward(&mut self, root: NodeId) {
        // Inicializa el gradiente del nodo raíz (usualmente la pérdida de forma [1])
        for g in &mut self.nodes[root.0].grad {
            *g = 1.0;
        }

        for i in (0..=root.0).rev() {
            let op = self.nodes[i].op;
            if matches!(op, Op::Leaf) { continue; }

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

                    unsafe {
                        // 1. dX = dZ * W
                        for i_m in 0..m {
                            for j_n in 0..n {
                                let dz_val = z.grad[i_m * n + j_n];
                                if dz_val == 0.0 { continue; }
                                // dX[i_m, :] += dz_val * W[j_n, :]
                                let x_grad_row = &mut x_node.grad[i_m * k .. (i_m + 1) * k];
                                let w_data_row = &w_node.data[j_n * k .. (j_n + 1) * k];
                                avx_math::axpy_avx2(x_grad_row, dz_val, w_data_row);
                            }
                        }

                        // 2. dW = dZ^T * X  =>  dW[j_n, :] += dZ[i_m, j_n] * X[i_m, :]
                        for i_m in 0..m {
                            for j_n in 0..n {
                                let dz_val = z.grad[i_m * n + j_n];
                                if dz_val == 0.0 { continue; }
                                let w_grad_row = &mut w_node.grad[j_n * k .. (j_n + 1) * k];
                                let x_data_row = &x_node.data[i_m * k .. (i_m + 1) * k];
                                avx_math::axpy_avx2(w_grad_row, dz_val, x_data_row);
                            }
                        }
                    }
                }
                Op::SiLU(x_id) => {
                    let z = self.nodes[i].clone();
                    let x_node = &mut self.nodes[x_id.0];
                    for j in 0..z.data.len() {
                        let dz = z.grad[j];
                        if dz == 0.0 { continue; }
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
                    
                    let max_logit = logits_node.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
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
                Op::Leaf => {}
            }
        }
    }

    // Helper functions for safe mutable aliasing split
    fn get_two(&self, id1: NodeId, id2: NodeId) -> (&Node, &Node) {
        (&self.nodes[id1.0], &self.nodes[id2.0])
    }

    // AG-01: get_two_mut/get_three_mut with bounds check to prevent UB.
    fn get_two_mut(&mut self, id1: NodeId, id2: NodeId) -> (&mut Node, &mut Node) {
        assert!(id1.0 < self.nodes.len() && id2.0 < self.nodes.len(), "NodeId out of bounds");
        assert!(id1.0 != id2.0);
        let ptr = self.nodes.as_mut_ptr();
        unsafe {
            (&mut *ptr.add(id1.0), &mut *ptr.add(id2.0))
        }
    }

    fn get_three_mut(&mut self, id1: NodeId, id2: NodeId, id3: NodeId) -> (&mut Node, &mut Node, &mut Node) {
        assert!(id1.0 < self.nodes.len() && id2.0 < self.nodes.len() && id3.0 < self.nodes.len(), "NodeId out of bounds");
        assert!(id1.0 != id2.0 && id1.0 != id3.0 && id2.0 != id3.0);
        let ptr = self.nodes.as_mut_ptr();
        unsafe {
            (&mut *ptr.add(id1.0), &mut *ptr.add(id2.0), &mut *ptr.add(id3.0))
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
        let w_data = vec![0.5, 0.5, 0.5, 
                          1.0, 1.0, 1.0];
        let w = tape.push_leaf(w_data, vec![2, 3]);

        // Z = X * W^T => [1, 2]
        let z = tape.linear(x, w);
        
        // Z[0] = 1*0.5 + 2*0.5 + 3*0.5 = 3.0
        // Z[1] = 1*1 + 2*1 + 3*1 = 6.0
        assert_eq!(tape.nodes[z.0].data, vec![3.0, 6.0]);

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
}
