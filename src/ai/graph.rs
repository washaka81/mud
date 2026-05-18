use std::collections::HashMap;
use crate::ai::store::MudStore;

/// A node in the MUD Knowledge Graph.
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

impl MudKnowledgeGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            content_to_index: HashMap::new(),
            bridge_threshold: 0.85,
            memory_limit: 1000,
        }
    }

    /// Adds a node and attempts to automatically bridge it to similar nodes.
    /// Manages memory by pruning low-rank nodes if limit reached.
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

        // Algorithm of Bridges: Connect to existing similar nodes
        for i in 0..self.nodes.len() {
            let sim = cosine_similarity(&embedding, &self.nodes[i].embedding);
            if sim > self.bridge_threshold {
                new_node.edges.push((i, sim));
                self.nodes[i].edges.push((id, sim));
            }
        }

        self.content_to_index.insert(content, id);
        self.nodes.push(new_node);

        // Avoid Memory Collapse: Prune if above limit
        if self.nodes.len() > self.memory_limit {
            self.prune();
        }
    }

    /// Simplified PageRank implementation to identify "hubs" of knowledge.
    pub fn recalculate_ranks(&mut self) {
        let n = self.nodes.len();
        if n == 0 { return; }
        
        let damping = 0.85f32;
        let base_rank = (1.0 - damping) / n as f32;

        for _ in 0..3 { // Fewer iterations for speed
            let mut new_ranks = vec![base_rank; n];
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
    let (mut dot, mut na, mut nb) = (0.0, 0.0, 0.0);
    let len = a.len().min(b.len());
    if len == 0 { return 0.0; }
    for i in 0..len {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-9)
}
