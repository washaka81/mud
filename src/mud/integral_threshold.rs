pub fn compute_optimal_thresholds(weights: &[f32]) -> (f32, f32) {
    if weights.is_empty() {
        return (1e-8, 0.0);
    }

    let mut max_abs = 0.0f32;
    for &w in weights {
        let abs = w.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }

    if max_abs < 1e-8 {
        return (1e-8, 0.0);
    }

    let num_bins = 100;
    let step = max_abs / num_bins as f32;

    let mut best_mse = f32::MAX;
    let mut best_delta = 0.0;
    let mut best_scale = 1e-8;

    for i in 1..num_bins {
        let delta = i as f32 * step;

        let mut sum_above = 0.0;
        let mut count_above = 0;

        for &w in weights {
            if w.abs() > delta {
                sum_above += w.abs();
                count_above += 1;
            }
        }

        let scale = if count_above > 0 {
            sum_above / count_above as f32
        } else {
            1e-8
        };

        let mut current_mse = 0.0;
        for &w in weights {
            let abs_w = w.abs();
            if abs_w <= delta {
                current_mse += w * w;
            } else {
                let diff = abs_w - scale;
                current_mse += diff * diff;
            }
        }

        if current_mse < best_mse {
            best_mse = current_mse;
            best_delta = delta;
            best_scale = scale;
        }
    }

    (best_scale.max(1e-8), best_delta)
}
