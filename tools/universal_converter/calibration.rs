use safetensors::SafeTensors;
use std::collections::HashMap;

/// Computes static magnitude dampening scales using a depth-based heuristic.
/// This mitigates the LogitVar explosion by artificially dampening the scales of deeper layers.
pub fn compute_scales(tensors: &SafeTensors) -> HashMap<String, f32> {
    let mut scales = HashMap::new();
    
    // First, find the maximum layer number
    let mut max_layer = 0;
    for (name, _) in tensors.tensors() {
        if name.contains("layers.") || name.contains("blk.") {
            let parts: Vec<&str> = name.split('.').collect();
            for p in parts {
                if let Ok(l) = p.parse::<usize>() {
                    if l > max_layer { max_layer = l; }
                }
            }
        }
    }
    
    // Total layers for normalization
    let total_layers = (max_layer + 1) as f32;
    
    // Generate dampening factors
    for (name, _) in tensors.tensors() {
        let mut dampening = 1.0f32;
        if name.contains("layers.") || name.contains("blk.") {
            let parts: Vec<&str> = name.split('.').collect();
            for p in parts {
                if let Ok(l) = p.parse::<usize>() {
                    // Heuristic: Deeper layers get more dampening. Max dampening is 0.65 (reduces by 35%)
                    let depth_ratio = (l as f32) / total_layers;
                    // Exponential dampening curve
                    dampening = 1.0 - (0.35 * depth_ratio.powi(2));
                    break;
                }
            }
        }
        scales.insert(name, dampening);
    }
    
    // Inject global metadata values as special keys
    scales.insert("__meta_max_layer".to_string(), max_layer as f32);
    
    scales
}
