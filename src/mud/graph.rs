use std::collections::HashMap;
use crate::mud::store::MudStore;

/// A node in the MUD Knowledge Graph.
#[derive(Clone)]
pub struct GraphNode {
    pub id: usize,
    pub content: String,
    pub embedding: Vec<f32>,
    /// Connections to other nodes (target_id, weight/relevance)
    pub edges: Vec<(usize, f32)>,
    /// PageRank-like score for importance
    pub rank: f32,
}

/// The MUD Knowledge Graph (MKG)
/// Allows "jumping" between related concepts and autonomous feedback.
pub struct MudKnowledgeGraph {
    pub nodes: Vec<GraphNode>,
    /// Index for fast lookup by content (prevents duplicates in RAM)
    pub content_to_index: HashMap<String, usize>,
    /// Threshold for autonomous bridging
    pub bridge_threshold: f32,
    /// Memory limit (max nodes in RAM)
    pub memory_limit: usize,
}

impl Default for MudKnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl MudKnowledgeGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            content_to_index: HashMap::new(),
            bridge_threshold: 0.85,
            memory_limit: 1000,
        }
    }

    /// Adds a node and automatically builds a "Synapse Mesh" (direct and indirect connections).
    /// Now rewards nodes with high connectivity to elevate their rank.
    pub fn add_node(&mut self, content: String, embedding: Vec<f32>) {
        if self.content_to_index.contains_key(&content) { return; }
        
        let id = self.nodes.len();
        let mut new_node = GraphNode {
            id,
            content: content.clone(),
            embedding: embedding.clone(),
            edges: Vec::new(),
            rank: 1.0,
        };

        // 1. Primary Synapses: Direct semantic similarity
        for i in 0..self.nodes.len() {
            let sim = cosine_similarity(&embedding, &self.nodes[i].embedding);
            if sim > self.bridge_threshold {
                new_node.edges.push((i, sim));
                self.nodes[i].edges.push((id, sim));
                
                // 2. Secondary Synapses: Logic Jumps
                // Connect to neighbors of neighbors to create a dense mesh
                let neighbor_edges = self.nodes[i].edges.clone();
                for &(neighbor_id, neighbor_sim) in &neighbor_edges {
                    if neighbor_id != id {
                        let indirect_sim = sim * neighbor_sim * 0.5; // Damped connection
                        if indirect_sim > 0.4 {
                             new_node.edges.push((neighbor_id, indirect_sim));
                        }
                    }
                }
            }
        }

        self.content_to_index.insert(content, id);
        self.nodes.push(new_node);

        // 3. Connectivity Reward: Boost rank based on synapse density
        self.apply_connectivity_reward(id);

        if self.nodes.len() > self.memory_limit {
            self.prune();
        }
    }

    /// Rewards a node by increasing its rank based on the number of synapses (connections) it holds.
    fn apply_connectivity_reward(&mut self, id: usize) {
        let connectivity = self.nodes[id].edges.len() as f32;
        // Reward formula: log scale boost to prevent runaway inflation
        let boost = (connectivity + 1.0).ln() * 0.1;
        self.nodes[id].rank += boost;
    }

    /// Enhanced PageRank that considers synapse weights for more accurate hub identification.
    pub fn recalculate_ranks(&mut self) {
        let n = self.nodes.len();
        if n == 0 { return; }

        let damping = 0.85f32;
        let base_rank = (1.0 - damping) / n as f32;

        // Initialize ranks uniformly
        for node in self.nodes.iter_mut() {
            node.rank = 1.0 / n as f32;
        }

        for _ in 0..3 {
            let mut new_ranks = vec![base_rank; n];
            // Handle dangling nodes: distribute their rank equally to all nodes
            let mut dangling_sum = 0.0f32;
            for i in 0..n {
                if self.nodes[i].edges.is_empty() {
                    dangling_sum += damping * self.nodes[i].rank / n as f32;
                }
            }
            if dangling_sum > 0.0 {
                for rank in new_ranks.iter_mut() {
                    *rank += dangling_sum;
                }
            }

            for i in 0..n {
                let edges = &self.nodes[i].edges;
                if edges.is_empty() { continue; }
                let share = damping * self.nodes[i].rank / edges.len() as f32;
                for &(target_id, _) in edges {
                    new_ranks[target_id] += share;
                }
            }
            for i in 0..n { self.nodes[i].rank = new_ranks[i]; }
        }
    }

    /// Prunes the graph to stay within memory limits, keeping high-rank hubs.
    fn prune(&mut self) {
        // Keep 80% of memory limit, removing lowest rank nodes
        let mut sorted_indices: Vec<usize> = (0..self.nodes.len()).collect();
        sorted_indices.sort_by(|&a, &b| self.nodes[b].rank.partial_cmp(&self.nodes[a].rank).unwrap());
        
        let keep_count = (self.memory_limit as f32 * 0.8) as usize;
        let to_keep: std::collections::HashSet<usize> = sorted_indices.iter().take(keep_count).cloned().collect();

        let mut new_nodes = Vec::new();
        let mut old_to_new = HashMap::new();

        for (i, node) in self.nodes.drain(..).enumerate() {
            if to_keep.contains(&i) {
                old_to_new.insert(i, new_nodes.len());
                new_nodes.push(node);
            }
        }

        // Update edges to new indices
        for node in &mut new_nodes {
            node.edges = node.edges.iter()
                .filter(|(tid, _)| to_keep.contains(tid))
                .map(|(tid, sim)| (*old_to_new.get(tid).unwrap(), *sim))
                .collect();
        }

        self.nodes = new_nodes;
        self.content_to_index.clear();
        for (i, node) in self.nodes.iter().enumerate() {
            self.content_to_index.insert(node.content.clone(), i);
        }
    }

    /// Performs an autonomous search, loading from disk if necessary.
    pub fn autonomous_jump_search(&mut self, query_vec: &[f32], store: &MudStore, depth: usize) -> Vec<String> {
        let mut results = Vec::new();
        let mut visited = std::collections::HashSet::new();

        let mut entry_idx = None;
        let mut max_sim = -1.0f32;
        for i in 0..self.nodes.len() {
            let sim = cosine_similarity(query_vec, &self.nodes[i].embedding);
            if sim > max_sim {
                max_sim = sim;
                entry_idx = Some(i);
            }
        }

        // Autonomous Disk Loading: If RAM doesn't have good matches, query SQLite
        if max_sim < 0.6 {
            if let Ok(candidates) = store.get_potential_candidates() {
                for (content, emb, rank) in candidates {
                    self.add_node(content, emb);
                    // Update rank from disk
                    if let Some(&idx) = self.content_to_index.get(&self.nodes.last().unwrap().content) {
                         self.nodes[idx].rank = rank;
                    }
                }
                // Re-evaluate entry point
                for i in 0..self.nodes.len() {
                    let sim = cosine_similarity(query_vec, &self.nodes[i].embedding);
                    if sim > max_sim {
                        max_sim = sim;
                        entry_idx = Some(i);
                    }
                }
            }
        }

        if let Some(idx) = entry_idx {
            if max_sim > 0.65 {
                self.traverse(idx, depth, &mut visited, &mut results);
            }
        }
        results
    }

    fn traverse(&self, idx: usize, depth: usize, visited: &mut std::collections::HashSet<usize>, out: &mut Vec<String>) {
        if depth == 0 || visited.contains(&idx) { return; }
        visited.insert(idx);
        out.push(self.nodes[idx].content.clone());

        let mut edges = self.nodes[idx].edges.clone();
        edges.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (target_id, _) in edges {
            self.traverse(target_id, depth - 1, visited, out);
        }
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 { return 0.0; }
    
    // Use AVX2 optimized kernels
    unsafe {
        let dot = crate::asm::dot_product_avx2(n, a.as_ptr(), b.as_ptr());
        let sa = crate::asm::sum_squares_avx2(n, a.as_ptr());
        let sb = crate::asm::sum_squares_avx2(n, b.as_ptr());
        dot / (sa.sqrt() * sb.sqrt() + 1e-9)
    }
}
