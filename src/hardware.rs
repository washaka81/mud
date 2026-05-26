use sysinfo::System;

#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub cpu_brand: String,
    pub total_cores: usize,
    pub p_cores: usize,
    pub e_cores: usize,
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub l3_cache_kb: u64,
    pub is_intel_hybrid: bool,
    pub preferred_threads: usize,
    pub vlk_device_name: String,
    pub vlk_is_integrated: bool,
}

impl HardwareProfile {
    pub fn detect() -> Self {
        let mut sys = System::new_all();
        sys.refresh_cpu_all();
        
        let cpus = sys.cpus();
        let cpu_brand = cpus.get(0).map(|c| c.brand().to_string()).unwrap_or_else(|| "Unknown CPU".to_string());
        let total_cores = cpus.len();
        
        let p_cores;
        let e_cores;
        let is_intel_hybrid = cpu_brand.contains("Intel") && (cpu_brand.contains("12") || cpu_brand.contains("13") || cpu_brand.contains("14") || cpu_brand.contains("Core"));
        
        if is_intel_hybrid {
            // Intel 1260P: 4 P-cores (8 threads) + 8 E-cores (8 threads) = 16 threads.
            if cpu_brand.contains("1260P") {
                p_cores = 8; 
                e_cores = 8;
            } else if total_cores >= 12 {
                // Heuristic for mobile hybrid i7/i5
                p_cores = 8; 
                e_cores = total_cores - 8;
            } else {
                p_cores = total_cores;
                e_cores = 0;
            }
        } else {
            p_cores = total_cores;
            e_cores = 0;
        }

        let has_avx2 = is_x86_feature_detected!("avx2");
        let has_avx512 = is_x86_feature_detected!("avx512f");
        
        // Preferred threads: target physical P-cores or a reasonable portion of high-perf cores
        let preferred_threads = if is_intel_hybrid { p_cores / 2 } else { total_cores / 2 };
        let preferred_threads = preferred_threads.max(4).min(total_cores);

        Self {
            cpu_brand,
            total_cores,
            p_cores,
            e_cores,
            has_avx2,
            has_avx512,
            l3_cache_kb: 18432, 
            is_intel_hybrid,
            preferred_threads,
            vlk_device_name: String::new(),
            vlk_is_integrated: false,
        }
    }
}
