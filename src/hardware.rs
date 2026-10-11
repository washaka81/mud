use sysinfo::System;

#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub cpu_brand: String,
    pub total_cores: usize,
    pub p_cores: usize,
    pub e_cores: usize,
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_bmi2: bool,
    pub l1_cache_kb: u64,
    pub l2_cache_kb: u64,
    pub l3_cache_kb: u64,
    pub total_ram_mb: u64,
    pub is_intel_hybrid: bool,
    pub preferred_threads: usize,
    pub vlk_device_name: String,
    pub vlk_is_integrated: bool,
}

impl HardwareProfile {
    pub fn detect() -> Self {
        let mut sys = System::new_all();
        sys.refresh_cpu_all();
        sys.refresh_memory();

        let cpus = sys.cpus();
        let cpu_brand = cpus
            .first()
            .map(|c| c.brand().to_string())
            .unwrap_or_else(|| "Unknown CPU".to_string());
        let total_cores = cpus.len();

        let p_cores;
        let e_cores;
        let is_intel_hybrid = cpu_brand.contains("Intel")
            && (cpu_brand.contains("12")
                || cpu_brand.contains("13")
                || cpu_brand.contains("14")
                || cpu_brand.contains("Core"));

        if is_intel_hybrid {
            if cpu_brand.contains("1260P") {
                p_cores = 8;
                e_cores = 8;
            } else if total_cores >= 12 {
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
        let has_bmi2 = is_x86_feature_detected!("bmi2");

        // Memory-bound workload: all logical cores perform equally for stream-like access.
        // Benchmark validated that total_cores (all logical CPUs) outperforms half-cores by 13% on i7-1260P.
        let preferred_threads = total_cores;
        let preferred_threads = preferred_threads.max(4).min(total_cores);
        // Env override: MUD_PCORE_THREADS (preferred) or legacy RAYON_NUM_THREADS
        let preferred_threads = std::env::var("MUD_PCORE_THREADS")
            .or_else(|_| std::env::var("RAYON_NUM_THREADS"))
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(preferred_threads)
            .clamp(1, 64);

        let total_ram_mb = sys.total_memory() / 1024 / 1024;

        let mut l1_cache_kb = 32 * (total_cores as u64);
        let mut l2_cache_kb = 256 * (total_cores as u64);
        let mut l3_cache_kb = 16384;

        // Linux-specific high-fidelity cache detection
        #[cfg(target_os = "linux")]
        {
            let read_cache_kb = |idx: u8| -> Option<u64> {
                let path = format!("/sys/devices/system/cpu/cpu0/cache/index{}/size", idx);
                std::fs::read_to_string(path).ok().and_then(|s| {
                    let s = s.trim().to_uppercase();
                    if s.ends_with('K') {
                        s[..s.len() - 1].parse::<u64>().ok()
                    } else if s.ends_with('M') {
                        s[..s.len() - 1].parse::<u64>().ok().map(|v| v * 1024)
                    } else {
                        None
                    }
                })
            };
            if let Some(l1) = read_cache_kb(0) {
                l1_cache_kb = l1 * (total_cores as u64);
            }
            if let Some(l2) = read_cache_kb(2) {
                l2_cache_kb = l2 * (total_cores as u64);
            }
            if let Some(l3) = read_cache_kb(3) {
                l3_cache_kb = l3;
            }
        }

        let mut vlk_device_name = "None".to_string();
        let mut vlk_is_integrated = false;

        if let Ok(entry) = unsafe { ash::Entry::load() } {
            if let Ok(instance) =
                unsafe { entry.create_instance(&ash::vk::InstanceCreateInfo::default(), None) }
            {
                if let Ok(devices) = unsafe { instance.enumerate_physical_devices() } {
                    if let Some(device) = devices.first() {
                        let props = unsafe { instance.get_physical_device_properties(*device) };
                        let name_cstr =
                            unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) };
                        vlk_device_name = name_cstr.to_string_lossy().into_owned();
                        vlk_is_integrated =
                            props.device_type == ash::vk::PhysicalDeviceType::INTEGRATED_GPU;
                    }
                }
            }
        }

        Self {
            cpu_brand,
            total_cores,
            p_cores,
            e_cores,
            has_avx2,
            has_avx512,
            has_bmi2,
            l1_cache_kb,
            l2_cache_kb,
            l3_cache_kb,
            total_ram_mb,
            is_intel_hybrid,
            preferred_threads,
            vlk_device_name,
            vlk_is_integrated,
        }
    }
}
