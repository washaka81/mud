//! # Phase 15: Bare-Metal Vulkan Backend (ash)
//!
//! Replaces the `vulkano` wrapper with direct `ash` bindings.
//!
//! ## P-00 Mandate: Raw Pointer Mastery
//! Every Vulkan object is a raw handle. No Arc<>, no internal lock-tracking.
//! Memory is managed via `gpu-allocator` (AAA industry standard).
//!
//! ## Double Buffering Architecture (EDGE-08 Mature)
//! Two `VkFence`s rotate between submissions:
//!   - Fence A guards Chunk N's backward pass on the GPU
//!   - CPU immediately starts Chunk N+1's forward pass
//!   - Before dispatching Chunk N+2, we wait on Fence A (which is already done)
//!     This achieves true CPU/GPU overlap with zero vulkano lock overhead.
//!
//! ## Shader Loading
//! SPIR-V is pre-compiled at build time and embedded via `include_bytes!`.
//! All shaders are the same `.comp` files — only the loader changes.

use ash::vk;
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use gpu_allocator::MemoryLocation;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// SPIR-V Shaders: pre-compiled at build time, embedded as raw bytes.
// Filenames match the existing .comp sources in assets/shaders/.
// ─────────────────────────────────────────────────────────────────────────────
mod spv {
    pub static TERNARY_GEMV: &[u8] =
        include_bytes!("../../assets/shaders/spirv/ternary_gemv_unified.spv");
    pub static SILU_GATE: &[u8] = include_bytes!("../../assets/shaders/spirv/silu_gate.spv");
    pub static SHADOW_OPTIM: &[u8] =
        include_bytes!("../../assets/shaders/spirv/shadow_optimizer.spv");
    pub static TERNARY_BACKWARD: &[u8] =
        include_bytes!("../../assets/shaders/spirv/ternary_backward.spv");
    pub static NEWTON_STEP1: &[u8] =
        include_bytes!("../../assets/shaders/spirv/newton_schulz_step1.spv");
    pub static NEWTON_STEP2: &[u8] =
        include_bytes!("../../assets/shaders/spirv/newton_schulz_step2.spv");
    pub static TELEMETRY: &[u8] =
        include_bytes!("../../assets/shaders/spirv/tensor_thermodynamics.spv");
    pub static HEARTBEAT: &[u8] = include_bytes!("../../assets/shaders/spirv/heartbeat.spv");
    // L-06: optional large-hidden / multi-head paths
    pub static RMS_NORM: &[u8] = include_bytes!("../../assets/shaders/spirv/rms_norm.spv");
    pub static MHA: &[u8] = include_bytes!("../../assets/shaders/spirv/mha.spv");
}

/// L-06: prefer GPU RMSNorm when hidden ≥ this (else CPU wins on dispatch overhead).
pub const RMS_GPU_MIN_HIDDEN: usize = 512;
/// L-06: prefer GPU MHA when seq_len * n_heads ≥ this.
pub const MHA_GPU_MIN_WORK: usize = 64;
/// Phase B: prefer GPU GEMV when `n_in * n_out` ≥ this (dispatch + UMA overhead).
pub const GEMV_GPU_MIN_WORK: usize = 256 * 256;

// ─────────────────────────────────────────────────────────────────────────────
// AshBuffer: A raw VRAM buffer with its allocation handle.
// Owns both the VkBuffer handle and the memory backing it.
// ─────────────────────────────────────────────────────────────────────────────
pub struct AshBuffer {
    pub handle: vk::Buffer,
    pub allocation: Allocation,
    /// If HOST_VISIBLE, this is the mapped pointer into VRAM (zero-copy).
    pub mapped_ptr: Option<*mut u8>,
    pub size_bytes: usize,
}

unsafe impl Send for AshBuffer {}
unsafe impl Sync for AshBuffer {}

impl AshBuffer {
    /// # Safety
    /// Raw write to VRAM via mapped pointer.
    /// Caller guarantees `data.len() * 4 <= self.size_bytes`.
    pub unsafe fn write_f32(&self, data: &[f32]) {
        let ptr = self.mapped_ptr.expect("Buffer is not host-visible") as *mut f32;
        // SAFETY: ptr is valid, size was checked at allocation.
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
    }

    /// # Safety
    /// Raw read from VRAM via mapped pointer.
    /// Caller guarantees `out.len() * 4 <= self.size_bytes`.
    pub unsafe fn read_f32(&self, out: &mut [f32]) {
        let ptr = self.mapped_ptr.expect("Buffer is not host-visible") as *const f32;
        // SAFETY: ptr is valid, size was checked at allocation.
        std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), out.len());
    }

    /// # Safety
    /// Raw write to VRAM via mapped pointer (u32 variant).
    pub unsafe fn write_u32(&self, data: &[u32]) {
        let ptr = self.mapped_ptr.expect("Buffer is not host-visible") as *mut u32;
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline descriptor: groups a VkPipeline + VkPipelineLayout + VkDescriptorSetLayout
// ─────────────────────────────────────────────────────────────────────────────
struct AshPipeline {
    handle: vk::Pipeline,
    layout: vk::PipelineLayout,
    desc_layout: vk::DescriptorSetLayout,
}

// ─────────────────────────────────────────────────────────────────────────────
// L-05: True double-buffer — 2 command buffers + 2 fences
// While GPU runs slot A, CPU records slot B. Wait only before reusing a slot.
// ─────────────────────────────────────────────────────────────────────────────
struct DoubleFrame {
    fences: [vk::Fence; 2],
    cmds: [vk::CommandBuffer; 2],
    /// Next slot to acquire (0 or 1).
    slot: usize,
}

impl DoubleFrame {
    /// # SAFETY: device + cmd_pool must be valid.
    unsafe fn new(device: &ash::Device, cmd_pool: vk::CommandPool) -> anyhow::Result<Self> {
        let info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let a = device.create_fence(&info, None)?;
        let b = device.create_fence(&info, None)?;
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(2);
        let cmds = device.allocate_command_buffers(&alloc)?;
        Ok(Self {
            fences: [a, b],
            cmds: [cmds[0], cmds[1]],
            slot: 0,
        })
    }

    /// Wait until this slot's previous submit is done; return a reset command buffer.
    /// # SAFETY: device must match the creating device.
    unsafe fn acquire(&mut self, device: &ash::Device) -> anyhow::Result<vk::CommandBuffer> {
        let f = self.fences[self.slot];
        device.wait_for_fences(&[f], true, u64::MAX)?;
        device.reset_fences(&[f])?;
        let cmd = self.cmds[self.slot];
        device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        Ok(cmd)
    }

    /// Submit the already-recorded buffer for the current slot and advance.
    /// # SAFETY: cmd must be the one returned by acquire for this slot and ended.
    unsafe fn submit(&mut self, device: &ash::Device, queue: vk::Queue) -> anyhow::Result<()> {
        let cmd = self.cmds[self.slot];
        let fence = self.fences[self.slot];
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
        device.queue_submit(queue, &[submit], fence)?;
        self.slot = 1 - self.slot;
        Ok(())
    }

    /// Block until both slots are idle (full GPU drain).
    unsafe fn wait_all(&self, device: &ash::Device) -> anyhow::Result<()> {
        device.wait_for_fences(&self.fences, true, u64::MAX)?;
        Ok(())
    }

    fn slot(&self) -> usize {
        self.slot
    }

    unsafe fn destroy(&self, device: &ash::Device) {
        for &f in &self.fences {
            device.destroy_fence(f, None);
        }
        // command buffers freed with pool
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AshContext: The bare-metal equivalent of VulkanContext.
// All fields are raw Vulkan handles. No Arc, no Mutex, no overhead.
// ─────────────────────────────────────────────────────────────────────────────
pub struct AshContext {
    // Raw Vulkan handles (P-00: the silicon itself)
    // Note: entry, physical, queue_family are stored for sub-allocator re-init and device queries.
    #[allow(dead_code)]
    entry: ash::Entry,
    instance: ash::Instance,
    #[allow(dead_code)]
    physical: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    #[allow(dead_code)]
    queue_family: u32,

    // Descriptor Pool — RESET flag allows recycling without per-set free overhead
    desc_pool: vk::DescriptorPool,

    cmd_pool: vk::CommandPool,

    // gpu-allocator: handles VkDeviceMemory sub-allocation (industry standard)
    allocator: Option<Allocator>,

    // Pipelines — one per shader
    gemv_pipe: AshPipeline,
    silu_pipe: AshPipeline,
    optimizer_pipe: AshPipeline,
    backward_pipe: AshPipeline,
    ns_step1_pipe: AshPipeline,
    ns_step2_pipe: AshPipeline,
    telemetry_pipe: AshPipeline,
    heartbeat_pipe: AshPipeline,
    /// L-06
    rms_norm_pipe: AshPipeline,
    mha_pipe: AshPipeline,

    // L-05: dual CB + dual fence (true double-buffer)
    frame: DoubleFrame,

    // Persistent VRAM buffer cache (key = tensor name, e.g. "blk.0.q")
    buffer_cache: HashMap<String, AshBuffer>,

    pub available: bool,
}

impl AshContext {
    /// Initialize the bare-metal Vulkan context.
    /// Selects the best available device (discrete > integrated > cpu).
    pub fn new() -> anyhow::Result<Self> {
        let use_vlk = std::env::var("MUD_USE_VULKAN").unwrap_or_else(|_| "1".to_string());
        if use_vlk == "0" || use_vlk.to_lowercase() == "false" {
            anyhow::bail!("Vulkan desactivado por MUD_USE_VULKAN");
        }

        // SAFETY: Loading the system Vulkan loader. Will panic if libvulkan.so is missing.
        let entry = unsafe { ash::Entry::load()? };

        let app_name = c"MUD";
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(0)
            .engine_name(c"Forge")
            .api_version(vk::API_VERSION_1_2);

        let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        // SAFETY: entry is valid, instance_info is stack-allocated and correct.
        let instance = unsafe { entry.create_instance(&instance_info, None)? };

        // ── Pick physical device ───────────────────────────────────────────
        // SAFETY: instance is valid.
        let physicals = unsafe { instance.enumerate_physical_devices()? };
        let physical = physicals
            .into_iter()
            .min_by_key(|&pd| unsafe {
                match instance.get_physical_device_properties(pd).device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 0u32,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                    vk::PhysicalDeviceType::CPU => 3,
                    _ => 4,
                }
            })
            .ok_or_else(|| anyhow::anyhow!("No hay ningún dispositivo Vulkan disponible"))?;

        // ── Queue family: pick the first COMPUTE queue ─────────────────────
        let queue_family = unsafe {
            instance
                .get_physical_device_queue_family_properties(physical)
                .iter()
                .enumerate()
                .find(|(_, p)| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|(i, _)| i as u32)
                .ok_or_else(|| anyhow::anyhow!("No hay cola COMPUTE"))?
        };

        let queue_prio = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&queue_prio);

        // Enable subgroup arithmetic (needed by shadow_optimizer.comp subgroupAdd)
        let mut subgroup_features =
            vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
        let mut subgroup_size_ctl =
            vk::PhysicalDeviceSubgroupSizeControlFeatures::default().compute_full_subgroups(true);

        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .push_next(&mut subgroup_features)
            .push_next(&mut subgroup_size_ctl);

        // SAFETY: physical, instance are valid.
        let device = unsafe { instance.create_device(physical, &device_info, None)? };
        // SAFETY: device is valid, queue_family and index 0 are confirmed above.
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        // ── gpu-allocator ───────────────────────────────────────────────────
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: physical,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?;

        // ── Command pool ────────────────────────────────────────────────────
        let cmd_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: device is valid.
        let cmd_pool = unsafe { device.create_command_pool(&cmd_pool_info, None)? };

        // ── Descriptor pool (RESET strategy) ───────────────────────────────
        // FREE_DESCRIPTOR_SET omitted — we reset the whole pool between steps.
        // This is 3-5x faster than individual set frees on Iris Xe.
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1024,
        }];
        let desc_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(256)
            .pool_sizes(&pool_sizes);
        // SAFETY: device is valid, pool_sizes are stack-allocated correctly.
        let desc_pool = unsafe { device.create_descriptor_pool(&desc_pool_info, None)? };

        // ── Pipelines ────────────────────────────────────────────────────────
        let gemv_pipe = Self::create_pipeline(&device, spv::TERNARY_GEMV, 4, &[push_range(28)])?;
        let silu_pipe = Self::create_pipeline(&device, spv::SILU_GATE, 3, &[push_range(8)])?;
        let optimizer_pipe =
            Self::create_pipeline(&device, spv::SHADOW_OPTIM, 4, &[push_range(20)])?;
        let backward_pipe =
            Self::create_pipeline(&device, spv::TERNARY_BACKWARD, 3, &[push_range(12)])?;
        let ns_step1_pipe = Self::create_pipeline(&device, spv::NEWTON_STEP1, 2, &[push_range(8)])?;
        let ns_step2_pipe = Self::create_pipeline(&device, spv::NEWTON_STEP2, 3, &[push_range(8)])?;
        let telemetry_pipe = Self::create_pipeline(&device, spv::TELEMETRY, 2, &[push_range(4)])?;
        let heartbeat_pipe = Self::create_pipeline(&device, spv::HEARTBEAT, 1, &[push_range(0)])?;
        // L-06: rms_norm push = 16 bytes (u32,u32,f32,u32); mha push = 16 bytes (4×u32)
        let rms_norm_pipe = Self::create_pipeline(&device, spv::RMS_NORM, 3, &[push_range(16)])?;
        let mha_pipe = Self::create_pipeline(&device, spv::MHA, 4, &[push_range(16)])?;

        // ── L-05: dual command buffers + dual fences ─────────────────────────
        let frame = unsafe { DoubleFrame::new(&device, cmd_pool)? };

        Ok(Self {
            entry,
            instance,
            physical,
            device,
            queue,
            queue_family,
            desc_pool,
            cmd_pool,
            allocator: Some(allocator),
            gemv_pipe,
            silu_pipe,
            optimizer_pipe,
            backward_pipe,
            ns_step1_pipe,
            ns_step2_pipe,
            telemetry_pipe,
            heartbeat_pipe,
            rms_norm_pipe,
            mha_pipe,
            frame,
            buffer_cache: HashMap::new(),
            available: true,
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Pipeline factory: SPIR-V → VkShaderModule → VkPipeline
    // n_bindings = number of storage buffers this shader reads/writes
    // push_ranges = list of push_constant ranges (usually 1)
    // ─────────────────────────────────────────────────────────────────────────
    fn create_pipeline(
        device: &ash::Device,
        spirv: &[u8],
        n_bindings: u32,
        push_ranges: &[vk::PushConstantRange],
    ) -> anyhow::Result<AshPipeline> {
        // 1. Load SPIR-V module safely by copying to an aligned Vec<u32>
        // include_bytes! gives &[u8] with 1-byte alignment, so we can't just cast it.
        let mut spv_u32 = vec![0u32; spirv.len() / 4];
        unsafe {
            std::ptr::copy_nonoverlapping(
                spirv.as_ptr(),
                spv_u32.as_mut_ptr() as *mut u8,
                spirv.len(),
            );
        }
        let shader_info = vk::ShaderModuleCreateInfo::default().code(&spv_u32);
        let shader_module = unsafe { device.create_shader_module(&shader_info, None)? };

        // 2. Descriptor set layout: one STORAGE_BUFFER binding per n_bindings
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..n_bindings)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: device is valid, bindings are stack-allocated.
        let desc_layout = unsafe { device.create_descriptor_set_layout(&dsl_info, None)? };

        // 3. Pipeline layout = descriptor layout + push constants
        let set_layouts = [desc_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(push_ranges);
        // SAFETY: device is valid.
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        // 4. Compute pipeline
        let entry_point = c"main";
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry_point);
        let pipe_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        let handle = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipe_info], None)
                .map_err(|(_, e)| anyhow::anyhow!("Pipeline creation failed: {e:?}"))?[0]
        };

        // SAFETY: shader module is no longer needed after pipeline creation.
        unsafe {
            device.destroy_shader_module(shader_module, None);
        }

        Ok(AshPipeline {
            handle,
            layout,
            desc_layout,
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Buffer Allocation
    // ─────────────────────────────────────────────────────────────────────────

    /// Allocate a HOST_VISIBLE + DEVICE_LOCAL buffer (zero-copy mapped).
    /// On Intel Iris Xe (UMA), this is real zero-copy into shared RAM.
    pub fn alloc_host_visible(
        &mut self,
        name: &str,
        size_bytes: usize,
    ) -> anyhow::Result<&AshBuffer> {
        if self.buffer_cache.contains_key(name) {
            return Ok(&self.buffer_cache[name]);
        }
        let buf = self.create_buffer_internal(name, size_bytes, MemoryLocation::CpuToGpu)?;
        self.buffer_cache.insert(name.to_string(), buf);
        Ok(&self.buffer_cache[name])
    }

    /// Allocate a DEVICE_LOCAL only buffer (fast VRAM, not CPU-readable).
    pub fn alloc_device_local(
        &mut self,
        name: &str,
        size_bytes: usize,
    ) -> anyhow::Result<&AshBuffer> {
        if self.buffer_cache.contains_key(name) {
            return Ok(&self.buffer_cache[name]);
        }
        let buf = self.create_buffer_internal(name, size_bytes, MemoryLocation::GpuOnly)?;
        self.buffer_cache.insert(name.to_string(), buf);
        Ok(&self.buffer_cache[name])
    }

    fn create_buffer_internal(
        &mut self,
        name: &str,
        size_bytes: usize,
        location: MemoryLocation,
    ) -> anyhow::Result<AshBuffer> {
        let buf_info = vk::BufferCreateInfo::default()
            .size(size_bytes as u64)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: device is valid, buf_info is valid.
        let handle = unsafe { self.device.create_buffer(&buf_info, None)? };

        // SAFETY: handle is a valid VkBuffer.
        let requirements = unsafe { self.device.get_buffer_memory_requirements(handle) };

        let allocation = self
            .allocator
            .as_mut()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;

        // SAFETY: handle and allocation.memory() are valid.
        unsafe {
            self.device
                .bind_buffer_memory(handle, allocation.memory(), allocation.offset())?;
        }

        let mapped_ptr =
            if location == MemoryLocation::CpuToGpu || location == MemoryLocation::GpuToCpu {
                allocation.mapped_ptr().map(|p| p.as_ptr() as *mut u8)
            } else {
                None
            };

        Ok(AshBuffer {
            handle,
            allocation,
            mapped_ptr,
            size_bytes,
        })
    }

    pub fn get_buffer(&self, name: &str) -> Option<&AshBuffer> {
        self.buffer_cache.get(name)
    }

    /// Ensure a HOST_VISIBLE buffer exists with at least `size_bytes`.
    /// Reallocates (destroy + create) if the cached buffer is too small.
    pub fn ensure_host_buffer(&mut self, name: &str, size_bytes: usize) -> anyhow::Result<()> {
        if let Some(buf) = self.buffer_cache.get(name) {
            if buf.size_bytes >= size_bytes {
                return Ok(());
            }
        }
        if let Some(old) = self.buffer_cache.remove(name) {
            unsafe {
                self.device.destroy_buffer(old.handle, None);
                if let Some(alloc) = self.allocator.as_mut() {
                    let _ = alloc.free(old.allocation);
                }
            }
        }
        self.alloc_host_visible(name, size_bytes)?;
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Command submission helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Allocate a single-use command buffer, record it, submit it, and block until done.
    /// Use for one-shot uploads and rarely-executed paths only.
    ///
    /// # SAFETY: device, cmd_pool, queue must be valid.
    unsafe fn submit_and_wait<F>(&self, record_fn: F) -> anyhow::Result<()>
    where
        F: FnOnce(vk::CommandBuffer) -> anyhow::Result<()>,
    {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = self.device.allocate_command_buffers(&alloc_info)?[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self.device.begin_command_buffer(cmd, &begin_info)?;
        record_fn(cmd)?;
        self.device.end_command_buffer(cmd)?;

        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
        let fence_info = vk::FenceCreateInfo::default();
        let fence = self.device.create_fence(&fence_info, None)?;
        self.device
            .queue_submit(self.queue, &[submit_info], fence)?;
        self.device.wait_for_fences(&[fence], true, u64::MAX)?;
        self.device.destroy_fence(fence, None);
        self.device.free_command_buffers(self.cmd_pool, &[cmd]);
        Ok(())
    }

    // submit_async removed — L-05 uses DoubleFrame::submit after acquire/record

    // ─────────────────────────────────────────────────────────────────────────
    // Descriptor set helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Build a descriptor set for `buffers` and bind it into `cmd`.
    ///
    /// # SAFETY: All VkBuffer handles inside buffers must be valid.
    unsafe fn bind_storage_buffers(
        &self,
        cmd: vk::CommandBuffer,
        pipe: &AshPipeline,
        buffers: &[vk::Buffer],
    ) -> anyhow::Result<()> {
        // Allocate descriptor set
        let layouts = [pipe.desc_layout];
        let dsa_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.desc_pool)
            .set_layouts(&layouts);
        let set = self.device.allocate_descriptor_sets(&dsa_info)?[0];

        // Write each buffer binding
        let buf_infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
            .iter()
            .map(|&b| {
                [vk::DescriptorBufferInfo {
                    buffer: b,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                }]
            })
            .collect();

        let writes: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();

        self.device.update_descriptor_sets(&writes, &[]);

        // Bind pipeline + descriptor set
        self.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe.handle);
        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            pipe.layout,
            0,
            &[set],
            &[],
        );

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PUBLIC DISPATCH API
    // Mirrors the vulkano VulkanContext interface so callers need minimal changes.
    // ─────────────────────────────────────────────────────────────────────────

    /// Run the QAT optimizer (SGD + PRQ) for all layer matrices in one command buffer.
    /// ASYNCHRONOUS: returns before the GPU finishes (EDGE-08 double-buffer).
    ///
    /// # SAFETY
    /// All buffer names must exist in self.buffer_cache.
    /// push_constants slices must have exactly the byte size the shader expects.
    pub unsafe fn dispatch_optimizer_batch_async(
        &mut self,
        updates: &[AshOptimizerUpdate],
    ) -> anyhow::Result<()> {
        // L-05: acquire free slot (wait only if that slot still in flight)
        let cmd = self.frame.acquire(&self.device)?;

        // Reset descriptor pool: reclaims all set memory in O(1).
        self.device
            .reset_descriptor_pool(self.desc_pool, vk::DescriptorPoolResetFlags::empty())?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self.device.begin_command_buffer(cmd, &begin_info)?;

        for u in updates {
            let shadow = self
                .buffer_cache
                .get(&u.shadow_key)
                .ok_or_else(|| anyhow::anyhow!("Missing shadow buffer: {}", u.shadow_key))?;
            let grad = self
                .buffer_cache
                .get(&u.grad_key)
                .ok_or_else(|| anyhow::anyhow!("Missing grad buffer: {}", u.grad_key))?;
            let scales = self
                .buffer_cache
                .get(&u.scales_key)
                .ok_or_else(|| anyhow::anyhow!("Missing scales buffer: {}", u.scales_key))?;
            let packed = self
                .buffer_cache
                .get(&u.packed_key)
                .ok_or_else(|| anyhow::anyhow!("Missing packed buffer: {}", u.packed_key))?;

            let bufs = [shadow.handle, grad.handle, scales.handle, packed.handle];
            self.bind_storage_buffers(cmd, &self.optimizer_pipe, &bufs)?;

            let pc = OptimizerPushConstants {
                total_elements: u.total_elements as u32,
                cols: u.cols as u32,
                learning_rate: u.learning_rate,
                weight_decay: u.weight_decay,
                num_tokens: u.num_tokens,
            };
            let pc_bytes = std::slice::from_raw_parts(
                &pc as *const _ as *const u8,
                std::mem::size_of::<OptimizerPushConstants>(),
            );
            self.device.cmd_push_constants(
                cmd,
                self.optimizer_pipe.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );

            let rows = u.total_elements / u.cols.max(1);
            self.device.cmd_dispatch(cmd, rows as u32, 1, 1);
        }

        self.device.end_command_buffer(cmd)?;
        // Submit async; other slot free for next record → true CPU/GPU overlap
        self.frame.submit(&self.device, self.queue)?;
        Ok(())
    }

    /// L-05: drain both double-buffer slots (call before readback / process exit).
    ///
    /// # Safety
    /// `device` must still be valid (context not dropped).
    pub unsafe fn sync_frames(&self) -> anyhow::Result<()> {
        self.frame.wait_all(&self.device)
    }

    /// L-05: which slot will be acquired next (0 or 1) — for tests / telemetry.
    pub fn frame_slot(&self) -> usize {
        self.frame.slot()
    }

    /// Phase B: host-visible ternary GEMV (upload → shader → readback).
    /// Prefer when `n_in * n_out` is large enough to amortize dispatch (see `GEMV_GPU_MIN_WORK`).
    ///
    /// # Safety
    /// `x.len() >= n_in`, `y.len() >= n_out`, `scales.len() >= n_out`,
    /// `packed` holds `n_out * (n_in/8)` u32 ELUT words. `n_in` multiple of 8.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dispatch_gemv_host_sync(
        &mut self,
        x: &[f32],
        packed: &[u32],
        scales: &[f32],
        y: &mut [f32],
        n_in: usize,
        n_out: usize,
        do_norm: bool,
    ) -> anyhow::Result<()> {
        self.dispatch_gemv_host_sync_ex(x, packed, scales, y, n_in, n_out, do_norm, true, true)
    }

    /// Phase B+: same as [`dispatch_gemv_host_sync`] with optional skip of weight/scale
    /// re-upload when the host pointers are unchanged (inference reuses tensor addresses).
    ///
    /// # Safety
    /// Same as [`dispatch_gemv_host_sync`]. When `upload_w`/`upload_sc` are false, VRAM
    /// buffers must still hold valid data from a prior call.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dispatch_gemv_host_sync_ex(
        &mut self,
        x: &[f32],
        packed: &[u32],
        scales: &[f32],
        y: &mut [f32],
        n_in: usize,
        n_out: usize,
        do_norm: bool,
        upload_w: bool,
        upload_sc: bool,
    ) -> anyhow::Result<()> {
        if !self.available {
            anyhow::bail!("AshContext unavailable");
        }
        if n_in == 0 || n_out == 0 || !n_in.is_multiple_of(8) {
            anyhow::bail!("gemv n_in must be >0 and multiple of 8");
        }
        let blocks = n_in / 8;
        if x.len() < n_in
            || y.len() < n_out
            || scales.len() < n_out
            || packed.len() < n_out * blocks
        {
            anyhow::bail!("gemv buffer size mismatch");
        }
        self.ensure_host_buffer("gemv_x", n_in * 4)?;
        self.ensure_host_buffer("gemv_w", n_out * blocks * 4)?;
        self.ensure_host_buffer("gemv_y", n_out * 4)?;
        self.ensure_host_buffer("gemv_sc", n_out * 4)?;
        // Activations always change per token
        self.buffer_cache["gemv_x"].write_f32(&x[..n_in]);
        if upload_w {
            let wbytes = n_out * blocks * 4;
            let src = std::slice::from_raw_parts(packed.as_ptr() as *const u8, wbytes);
            let dst = self.buffer_cache["gemv_w"]
                .mapped_ptr
                .expect("gemv_w host visible");
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, wbytes);
        }
        if upload_sc {
            self.buffer_cache["gemv_sc"].write_f32(&scales[..n_out]);
        }

        let xh = self.buffer_cache["gemv_x"].handle;
        let wh = self.buffer_cache["gemv_w"].handle;
        let yh = self.buffer_cache["gemv_y"].handle;
        let sh = self.buffer_cache["gemv_sc"].handle;
        let pipe = &self.gemv_pipe as *const AshPipeline;

        self.submit_and_wait(|cmd| {
            let pipe = &*pipe;
            self.bind_storage_buffers(cmd, pipe, &[xh, wh, yh, sh])?;
            let pc = GemvPushConstants {
                n_in: n_in as u32,
                n_out: n_out as u32,
                batch_size: 1,
                inv_q_scale: 1.0,
                do_norm: if do_norm { 1 } else { 0 },
                single_scale: 0,
                scale: 1.0,
            };
            let pc_bytes = std::slice::from_raw_parts(
                &pc as *const _ as *const u8,
                std::mem::size_of::<GemvPushConstants>(),
            );
            self.device.cmd_push_constants(
                cmd,
                pipe.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );
            self.device.cmd_dispatch(cmd, n_out as u32, 1, 1);
            Ok(())
        })?;
        self.buffer_cache["gemv_y"].read_f32(&mut y[..n_out]);
        Ok(())
    }

    /// Stream F: Q, K, V ternary GEMV in **one** command buffer (single fence).
    ///
    /// Uploads activations once; optionally re-uploads each weight/scale pair.
    /// Three compute dispatches share the same `x` (independent W/Y buffers).
    /// Prefer over three [`dispatch_gemv_host_sync_ex`] calls to amortize submit latency.
    ///
    /// # Safety
    /// - `x.len() >= n_in`
    /// - Q: `q_packed` / `q_scales` / `q_y` cover `n_q` rows × `n_in`
    /// - K,V: same with `n_kv`
    /// - `n_in` multiple of 8
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dispatch_gemv_qkv_host_sync(
        &mut self,
        x: &[f32],
        q_packed: &[u32],
        k_packed: &[u32],
        v_packed: &[u32],
        q_scales: &[f32],
        k_scales: &[f32],
        v_scales: &[f32],
        q_y: &mut [f32],
        k_y: &mut [f32],
        v_y: &mut [f32],
        n_in: usize,
        n_q: usize,
        n_kv: usize,
        upload_q: bool,
        upload_k: bool,
        upload_v: bool,
    ) -> anyhow::Result<()> {
        if !self.available {
            anyhow::bail!("AshContext unavailable");
        }
        if n_in == 0 || n_q == 0 || n_kv == 0 || !n_in.is_multiple_of(8) {
            anyhow::bail!("qkv gemv dims invalid");
        }
        let blocks = n_in / 8;
        if x.len() < n_in
            || q_y.len() < n_q
            || k_y.len() < n_kv
            || v_y.len() < n_kv
            || q_scales.len() < n_q
            || k_scales.len() < n_kv
            || v_scales.len() < n_kv
            || q_packed.len() < n_q * blocks
            || k_packed.len() < n_kv * blocks
            || v_packed.len() < n_kv * blocks
        {
            anyhow::bail!("qkv gemv buffer size mismatch");
        }

        self.ensure_host_buffer("gemv_x", n_in * 4)?;
        self.ensure_host_buffer("gemv_q_w", n_q * blocks * 4)?;
        self.ensure_host_buffer("gemv_k_w", n_kv * blocks * 4)?;
        self.ensure_host_buffer("gemv_v_w", n_kv * blocks * 4)?;
        self.ensure_host_buffer("gemv_q_sc", n_q * 4)?;
        self.ensure_host_buffer("gemv_k_sc", n_kv * 4)?;
        self.ensure_host_buffer("gemv_v_sc", n_kv * 4)?;
        self.ensure_host_buffer("gemv_q_y", n_q * 4)?;
        self.ensure_host_buffer("gemv_k_y", n_kv * 4)?;
        self.ensure_host_buffer("gemv_v_y", n_kv * 4)?;

        self.buffer_cache["gemv_x"].write_f32(&x[..n_in]);
        let copy_w = |name: &str, packed: &[u32], rows: usize| {
            let wbytes = rows * blocks * 4;
            let src = std::slice::from_raw_parts(packed.as_ptr() as *const u8, wbytes);
            let dst = self.buffer_cache[name].mapped_ptr.expect("host visible");
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, wbytes);
        };
        if upload_q {
            copy_w("gemv_q_w", q_packed, n_q);
            self.buffer_cache["gemv_q_sc"].write_f32(&q_scales[..n_q]);
        }
        if upload_k {
            copy_w("gemv_k_w", k_packed, n_kv);
            self.buffer_cache["gemv_k_sc"].write_f32(&k_scales[..n_kv]);
        }
        if upload_v {
            copy_w("gemv_v_w", v_packed, n_kv);
            self.buffer_cache["gemv_v_sc"].write_f32(&v_scales[..n_kv]);
        }

        let xh = self.buffer_cache["gemv_x"].handle;
        let qw = self.buffer_cache["gemv_q_w"].handle;
        let kw = self.buffer_cache["gemv_k_w"].handle;
        let vw = self.buffer_cache["gemv_v_w"].handle;
        let qs = self.buffer_cache["gemv_q_sc"].handle;
        let ks = self.buffer_cache["gemv_k_sc"].handle;
        let vs = self.buffer_cache["gemv_v_sc"].handle;
        let qy = self.buffer_cache["gemv_q_y"].handle;
        let ky = self.buffer_cache["gemv_k_y"].handle;
        let vy = self.buffer_cache["gemv_v_y"].handle;
        let pipe = &self.gemv_pipe as *const AshPipeline;

        // One CB: three dispatches, one fence (stream F amortization).
        self.submit_and_wait(|cmd| {
            let pipe = &*pipe;
            let record = |cmd: vk::CommandBuffer,
                          w: vk::Buffer,
                          y: vk::Buffer,
                          sc: vk::Buffer,
                          n_out: usize|
             -> anyhow::Result<()> {
                self.bind_storage_buffers(cmd, pipe, &[xh, w, y, sc])?;
                let pc = GemvPushConstants {
                    n_in: n_in as u32,
                    n_out: n_out as u32,
                    batch_size: 1,
                    inv_q_scale: 1.0,
                    do_norm: 0,
                    single_scale: 0,
                    scale: 1.0,
                };
                let pc_bytes = std::slice::from_raw_parts(
                    &pc as *const _ as *const u8,
                    std::mem::size_of::<GemvPushConstants>(),
                );
                self.device.cmd_push_constants(
                    cmd,
                    pipe.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    pc_bytes,
                );
                self.device.cmd_dispatch(cmd, n_out as u32, 1, 1);
                Ok(())
            };
            record(cmd, qw, qy, qs, n_q)?;
            // Memory barrier between dispatches (portable iGPU scheduling).
            let mem_barrier = || {
                vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            };
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[mem_barrier()],
                &[],
                &[],
            );
            record(cmd, kw, ky, ks, n_kv)?;
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[mem_barrier()],
                &[],
                &[],
            );
            record(cmd, vw, vy, vs, n_kv)?;
            Ok(())
        })?;

        self.buffer_cache["gemv_q_y"].read_f32(&mut q_y[..n_q]);
        self.buffer_cache["gemv_k_y"].read_f32(&mut k_y[..n_kv]);
        self.buffer_cache["gemv_v_y"].read_f32(&mut v_y[..n_kv]);
        Ok(())
    }

    /// Run the ternary GEMV (forward pass matrix multiply).
    /// SYNCHRONOUS — used by inference path where we need results immediately.
    ///
    /// # SAFETY
    /// All buffer names must exist in self.buffer_cache.
    pub unsafe fn dispatch_gemv_sync(
        &self,
        x_key: &str,
        w_key: &str,
        y_key: &str,
        sc_key: &str,
        n_in: usize,
        n_out: usize,
    ) -> anyhow::Result<()> {
        let x = self
            .buffer_cache
            .get(x_key)
            .ok_or_else(|| anyhow::anyhow!("Missing buffer: {x_key}"))?;
        let w = self
            .buffer_cache
            .get(w_key)
            .ok_or_else(|| anyhow::anyhow!("Missing buffer: {w_key}"))?;
        let y = self
            .buffer_cache
            .get(y_key)
            .ok_or_else(|| anyhow::anyhow!("Missing buffer: {y_key}"))?;
        let sc = self
            .buffer_cache
            .get(sc_key)
            .ok_or_else(|| anyhow::anyhow!("Missing buffer: {sc_key}"))?;

        let xh = x.handle;
        let wh = w.handle;
        let yh = y.handle;
        let sh = sc.handle;
        let pipe_ref = &self.gemv_pipe as *const AshPipeline;

        self.submit_and_wait(|cmd| {
            // SAFETY: all handles are valid, pipe_ref is valid for the lifetime of self.
            let pipe = &*pipe_ref;
            let bufs = [xh, wh, yh, sh];
            unsafe { self.bind_storage_buffers(cmd, pipe, &bufs)? };

            let pc = GemvPushConstants {
                n_in: n_in as u32,
                n_out: n_out as u32,
                batch_size: 1,
                inv_q_scale: 1.0,
                do_norm: 0,
                single_scale: 0,
                scale: 1.0,
            };
            let pc_bytes = unsafe {
                std::slice::from_raw_parts(&pc as *const _ as *const u8, std::mem::size_of_val(&pc))
            };
            unsafe {
                self.device.cmd_push_constants(
                    cmd,
                    pipe.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    pc_bytes,
                );
                self.device.cmd_dispatch(cmd, n_out as u32, 1, 1);
            }
            Ok(())
        })?;
        Ok(())
    }

    /// Block until the GPU finishes all outstanding async work (drain the pipeline).
    /// # Safety
    /// device and queue must be valid.
    pub unsafe fn sync(&self) -> anyhow::Result<()> {
        // L-05: wait both double-buffer slots first, then full device idle
        self.frame.wait_all(&self.device)?;
        self.device.device_wait_idle()?;
        Ok(())
    }

    /// L-06: RMSNorm y = x * inv_rms * w  (seq_len positions × hidden).
    /// Host-visible buffers: uploads `x` and `w`, runs shader, readbacks `y`.
    /// Prefer when `hidden >= RMS_GPU_MIN_HIDDEN` and Vulkan is available.
    ///
    /// # Safety
    /// `x.len() >= seq_len * hidden`, `w.len() >= hidden`, `y.len() >= seq_len * hidden`.
    pub unsafe fn dispatch_rms_norm_sync(
        &mut self,
        x: &[f32],
        w: &[f32],
        y: &mut [f32],
        hidden: usize,
        seq_len: usize,
        eps: f32,
    ) -> anyhow::Result<()> {
        if !self.available {
            anyhow::bail!("AshContext unavailable");
        }
        let n = hidden * seq_len;
        if x.len() < n || y.len() < n || w.len() < hidden {
            anyhow::bail!("rms_norm buffer size mismatch");
        }
        self.ensure_host_buffer("rms_x", n * 4)?;
        self.ensure_host_buffer("rms_w", hidden * 4)?;
        self.ensure_host_buffer("rms_y", n * 4)?;
        self.buffer_cache["rms_x"].write_f32(&x[..n]);
        self.buffer_cache["rms_w"].write_f32(&w[..hidden]);

        let xh = self.buffer_cache["rms_x"].handle;
        let wh = self.buffer_cache["rms_w"].handle;
        let yh = self.buffer_cache["rms_y"].handle;
        let pipe = &self.rms_norm_pipe as *const AshPipeline;

        self.submit_and_wait(|cmd| {
            let pipe = &*pipe;
            self.bind_storage_buffers(cmd, pipe, &[xh, wh, yh])?;
            let pc = RmsNormPushConstants {
                hidden_size: hidden as u32,
                seq_len: seq_len as u32,
                eps,
                reserved: 0,
            };
            let pc_bytes = std::slice::from_raw_parts(
                &pc as *const _ as *const u8,
                std::mem::size_of::<RmsNormPushConstants>(),
            );
            self.device.cmd_push_constants(
                cmd,
                pipe.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );
            self.device.cmd_dispatch(cmd, seq_len as u32, 1, 1);
            Ok(())
        })?;

        self.buffer_cache["rms_y"].read_f32(&mut y[..n]);
        Ok(())
    }

    /// L-06: Multi-head causal attention for short sequences.
    /// Layout: q [seq, n_head, head_dim], k/v [seq, n_kv_head, head_dim], out same as q.
    /// Prefer when `seq_len * n_head >= MHA_GPU_MIN_WORK`.
    ///
    /// # Safety
    /// Buffer lengths must match layout; seq_len ≤ 64 (shader shared scores[64]).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dispatch_mha_sync(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        seq_len: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
    ) -> anyhow::Result<()> {
        if !self.available {
            anyhow::bail!("AshContext unavailable");
        }
        if seq_len == 0 || seq_len > 64 {
            anyhow::bail!("mha seq_len must be 1..=64 (shader shared limit)");
        }
        let q_n = seq_len * n_head * head_dim;
        let kv_n = seq_len * n_kv_head * head_dim;
        if q.len() < q_n || out.len() < q_n || k.len() < kv_n || v.len() < kv_n {
            anyhow::bail!("mha buffer size mismatch");
        }
        self.ensure_host_buffer("mha_q", q_n * 4)?;
        self.ensure_host_buffer("mha_k", kv_n * 4)?;
        self.ensure_host_buffer("mha_v", kv_n * 4)?;
        self.ensure_host_buffer("mha_o", q_n * 4)?;
        self.buffer_cache["mha_q"].write_f32(&q[..q_n]);
        self.buffer_cache["mha_k"].write_f32(&k[..kv_n]);
        self.buffer_cache["mha_v"].write_f32(&v[..kv_n]);

        let qh = self.buffer_cache["mha_q"].handle;
        let kh = self.buffer_cache["mha_k"].handle;
        let vh = self.buffer_cache["mha_v"].handle;
        let oh = self.buffer_cache["mha_o"].handle;
        let pipe = &self.mha_pipe as *const AshPipeline;

        self.submit_and_wait(|cmd| {
            let pipe = &*pipe;
            self.bind_storage_buffers(cmd, pipe, &[qh, kh, vh, oh])?;
            let pc = MhaPushConstants {
                seq_len: seq_len as u32,
                n_head: n_head as u32,
                n_kv_head: n_kv_head as u32,
                head_dim: head_dim as u32,
            };
            let pc_bytes = std::slice::from_raw_parts(
                &pc as *const _ as *const u8,
                std::mem::size_of::<MhaPushConstants>(),
            );
            self.device.cmd_push_constants(
                cmd,
                pipe.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );
            // Dispatch [n_heads, seq_len, 1]
            self.device
                .cmd_dispatch(cmd, n_head as u32, seq_len as u32, 1);
            Ok(())
        })?;

        self.buffer_cache["mha_o"].read_f32(&mut out[..q_n]);
        Ok(())
    }

    /// Dispatches a tiny, fast (micros-scale) compute payload to prevent the GPU
    /// from entering RC6 / Deep Sleep power states during long CPU workloads.
    ///
    /// # Safety
    /// The caller must ensure the Vulkan device and queues are valid.
    pub unsafe fn dispatch_heartbeat_sync(&self) -> anyhow::Result<()> {
        if !self.available {
            return Ok(());
        }

        // Use any small buffer (like telemetry) to satisfy binding = 0
        let buf = match self.buffer_cache.values().next() {
            Some(b) => b.handle,
            None => return Ok(()),
        };

        self.submit_and_wait(|cmd| {
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.heartbeat_pipe.handle,
            );

            self.bind_storage_buffers(cmd, &self.heartbeat_pipe, &[buf])?;

            self.device.cmd_dispatch(cmd, 1, 1, 1);
            Ok(())
        })
    }

    pub fn is_available(&self) -> bool {
        self.available
    }

    /// L-02: Newton-Schulz orthogonalization on GPU (Iris Xe compute).
    /// Matches CPU `muon::newton_schulz_orthogonalize` inner loop (after norm):
    ///   tmp = XᵀX ; next = 1.5·X − 0.5·X·tmp  (n_iters times)
    ///
    /// Caller must normalize `x` by Frobenius norm before call and re-scale after.
    /// Synchronous — blocks until complete, then readbacks into `x`.
    ///
    /// # Safety
    /// `x.len() == rows * cols`. Device must remain valid for the call duration.
    pub unsafe fn dispatch_newton_schulz_sync(
        &mut self,
        x: &mut [f32],
        rows: usize,
        cols: usize,
        n_iters: usize,
    ) -> anyhow::Result<()> {
        if !self.available {
            anyhow::bail!("AshContext not available");
        }
        if rows == 0 || cols == 0 || n_iters == 0 {
            return Ok(());
        }
        if x.len() != rows * cols {
            anyhow::bail!(
                "NS size mismatch: x.len()={} rows*cols={}",
                x.len(),
                rows * cols
            );
        }

        let x_bytes = rows * cols * 4;
        let tmp_bytes = cols * cols * 4;
        self.ensure_host_buffer("ns_x", x_bytes)?;
        self.ensure_host_buffer("ns_tmp", tmp_bytes)?;
        self.ensure_host_buffer("ns_next", x_bytes)?;

        // Upload X (already normalized by caller)
        {
            let buf = self
                .buffer_cache
                .get("ns_x")
                .ok_or_else(|| anyhow::anyhow!("ns_x missing"))?;
            buf.write_f32(x);
        }

        let x_h = self.buffer_cache["ns_x"].handle;
        let tmp_h = self.buffer_cache["ns_tmp"].handle;
        let next_h = self.buffer_cache["ns_next"].handle;

        let pc = NsPushConstants {
            rows: rows as u32,
            cols: cols as u32,
        };
        let pc_bytes = std::slice::from_raw_parts(
            &pc as *const _ as *const u8,
            std::mem::size_of::<NsPushConstants>(),
        );

        let groups_c = cols.div_ceil(16) as u32;
        let groups_r = rows.div_ceil(16) as u32;

        // One command buffer: all NS iterations with compute barriers + buffer copies
        self.submit_and_wait(|cmd| {
            for _iter in 0..n_iters {
                // ── Step 1: tmp = X^T X  (cols × cols) ──
                self.bind_storage_buffers(cmd, &self.ns_step1_pipe, &[x_h, tmp_h])?;
                self.device.cmd_push_constants(
                    cmd,
                    self.ns_step1_pipe.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    pc_bytes,
                );
                self.device.cmd_dispatch(cmd, groups_c, groups_c, 1);

                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );

                // ── Step 2: next = 1.5 X - 0.5 X tmp ──
                self.bind_storage_buffers(cmd, &self.ns_step2_pipe, &[x_h, tmp_h, next_h])?;
                self.device.cmd_push_constants(
                    cmd,
                    self.ns_step2_pipe.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    pc_bytes,
                );
                self.device.cmd_dispatch(cmd, groups_c, groups_r, 1);

                // Copy next → x for next iteration
                let barrier2 = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::SHADER_READ);
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[barrier2],
                    &[],
                    &[],
                );

                let region = vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: x_bytes as u64,
                };
                self.device
                    .cmd_copy_buffer(cmd, next_h, x_h, std::slice::from_ref(&region));

                let barrier3 = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[barrier3],
                    &[],
                    &[],
                );
            }
            Ok(())
        })?;

        // Read back orthogonalized X
        {
            let buf = self
                .buffer_cache
                .get("ns_x")
                .ok_or_else(|| anyhow::anyhow!("ns_x missing after NS"))?;
            buf.read_f32(x);
        }

        Ok(())
    }
}

impl Drop for AshContext {
    fn drop(&mut self) {
        // SAFETY: We own all these handles; nothing else can use them after drop.
        unsafe {
            // Note: intentionally skip queue_wait_idle here — some Intel iGPU
            // drivers hang forever on teardown after compute dispatches, which
            // blocked `cargo test` process exit (L-02). Resources are still
            // destroyed; the kernel reclaims in-flight work on device destroy.

            // Destroy fences
            self.frame.destroy(&self.device);

            // Destroy buffers (must free allocation before destroying buffer)
            for (_, buf) in self.buffer_cache.drain() {
                self.device.destroy_buffer(buf.handle, None);
                if let Some(alloc) = self.allocator.as_mut() {
                    let _ = alloc.free(buf.allocation);
                }
            }

            // Destroy pipelines + layouts + descriptor set layouts
            for pipe in [
                &self.gemv_pipe,
                &self.silu_pipe,
                &self.optimizer_pipe,
                &self.backward_pipe,
                &self.ns_step1_pipe,
                &self.ns_step2_pipe,
                &self.telemetry_pipe,
                &self.heartbeat_pipe,
                &self.rms_norm_pipe,
                &self.mha_pipe,
            ] {
                self.device.destroy_pipeline(pipe.handle, None);
                self.device.destroy_pipeline_layout(pipe.layout, None);
                self.device
                    .destroy_descriptor_set_layout(pipe.desc_layout, None);
            }

            self.device.destroy_descriptor_pool(self.desc_pool, None);
            self.device.destroy_command_pool(self.cmd_pool, None);

            // Explicitly drop the allocator before destroying the device
            let _ = self.allocator.take();

            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Push constant structs (must match shader layout exactly — std430)
// ─────────────────────────────────────────────────────────────────────────────

#[repr(C)]
struct OptimizerPushConstants {
    total_elements: u32,
    cols: u32,
    learning_rate: f32,
    weight_decay: f32,
    num_tokens: f32,
}

#[repr(C)]
struct GemvPushConstants {
    n_in: u32,
    n_out: u32,
    batch_size: u32,
    inv_q_scale: f32,
    do_norm: u32,
    single_scale: u32,
    scale: f32,
}

/// Must match newton_schulz_step1/2.comp push constants (rows, cols).
#[repr(C)]
struct NsPushConstants {
    rows: u32,
    cols: u32,
}

#[repr(C)]
struct RmsNormPushConstants {
    hidden_size: u32,
    seq_len: u32,
    eps: f32,
    reserved: u32,
}

#[repr(C)]
struct MhaPushConstants {
    seq_len: u32,
    n_head: u32,
    n_kv_head: u32,
    head_dim: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API types
// ─────────────────────────────────────────────────────────────────────────────

/// Describes one matrix to be updated by `dispatch_optimizer_batch_async`.
pub struct AshOptimizerUpdate {
    pub shadow_key: String,
    pub grad_key: String,
    pub scales_key: String,
    pub packed_key: String,
    pub total_elements: usize,
    pub cols: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub num_tokens: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build a push constant range for a given byte size
// ─────────────────────────────────────────────────────────────────────────────
fn push_range(size_bytes: u32) -> vk::PushConstantRange {
    vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        offset: 0,
        size: if size_bytes == 0 { 4 } else { size_bytes }, // min 4 per spec
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
/// Lightweight probe: does a usable Vulkan (ash) compute device exist?
///
/// Returns `(available, description)`. Creates a throwaway `ash` instance,
/// enumerates physical devices and reports the preferred one's name + type
/// (e.g. `"Intel(R) Iris(R) Xe Graphics [integrated]"`). Honors
/// `MUD_USE_VULKAN=0`. Does NOT create a logical device — cheap enough for the
/// startup banner. All GPU compute in this project runs through `ash` (0.38);
/// there is no other Vulkan backend.
pub fn probe_gpu() -> (bool, String) {
    let use_vlk = std::env::var("MUD_USE_VULKAN").unwrap_or_else(|_| "1".to_string());
    if use_vlk == "0" || use_vlk.eq_ignore_ascii_case("false") {
        return (false, "disabled (MUD_USE_VULKAN=0)".to_string());
    }
    let entry = match unsafe { ash::Entry::load() } {
        Ok(e) => e,
        Err(_) => return (false, "no libvulkan loader".to_string()),
    };
    let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
    let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = match unsafe { entry.create_instance(&instance_info, None) } {
        Ok(i) => i,
        Err(_) => return (false, "no Vulkan instance".to_string()),
    };
    let chosen = (|| {
        let physicals = unsafe { instance.enumerate_physical_devices() }.ok()?;
        let physical = physicals.into_iter().min_by_key(|&pd| unsafe {
            match instance.get_physical_device_properties(pd).device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0u32,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                vk::PhysicalDeviceType::CPU => 3,
                _ => 4,
            }
        })?;
        let props = unsafe { instance.get_physical_device_properties(physical) };
        // device_name is a null-terminated i8 array.
        let name = {
            let bytes: Vec<u8> = props
                .device_name
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect();
            String::from_utf8_lossy(&bytes).trim().to_string()
        };
        let kind = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => "discrete",
            vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated",
            vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual",
            vk::PhysicalDeviceType::CPU => "cpu",
            _ => "other",
        };
        let has_compute = unsafe {
            instance
                .get_physical_device_queue_family_properties(physical)
                .iter()
                .any(|p| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
        };
        Some((has_compute, format!("{name} [{kind}] (ash 0.38)")))
    })();
    unsafe { instance.destroy_instance(None) };
    // Defense-in-depth: the Intel Iris Xe (ADL GT2) UMA driver SIGSEGVs inside
    // `submit_and_wait` when dispatching the QKV GEMV compute shader (uncatchable as
    // `Result`; deterministic at block 11/64). Never select it for compute — degrade to
    // AVX2 even if the user did not set MUD_USE_VULKAN=0 via mud.sh.
    let name_lower = chosen
        .as_ref()
        .map(|(_, s)| s.to_ascii_lowercase())
        .unwrap_or_default();
    if name_lower.contains("iris") && name_lower.contains("xe") {
        return (
            false,
            "blacklisted (crash-prone Intel Iris Xe driver)".to_string(),
        );
    }
    chosen.unwrap_or((false, "no compute device".to_string()))
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_gpu_no_panic() {
        // Must never panic regardless of whether a device exists.
        let (_avail, desc) = probe_gpu();
        assert!(!desc.is_empty());
    }

    #[test]
    fn test_push_range_min_size() {
        let r = push_range(0);
        assert_eq!(
            r.size, 4,
            "Push constant range must be at least 4 bytes per Vulkan spec"
        );
    }

    #[test]
    fn test_push_range_normal() {
        let r = push_range(16);
        assert_eq!(r.size, 16);
        assert_eq!(r.offset, 0);
        assert_eq!(r.stage_flags, vk::ShaderStageFlags::COMPUTE);
    }

    #[test]
    fn test_ash_context_disabled() {
        // When MUD_USE_VULKAN=0, should bail gracefully
        unsafe {
            std::env::set_var("MUD_USE_VULKAN", "0");
        }
        let result = AshContext::new();
        unsafe {
            std::env::set_var("MUD_USE_VULKAN", "1");
        }
        assert!(
            result.is_err(),
            "Should fail gracefully when Vulkan is disabled"
        );
    }

    #[test]
    fn test_push_constant_sizes_l06() {
        assert_eq!(std::mem::size_of::<RmsNormPushConstants>(), 16);
        assert_eq!(std::mem::size_of::<MhaPushConstants>(), 16);
        assert_eq!(std::mem::size_of::<OptimizerPushConstants>(), 20);
    }

    #[test]
    fn test_l05_double_frame_slot_rotates() {
        // Skip if no GPU / Vulkan disabled in CI
        let Ok(mut ctx) = AshContext::new() else {
            return;
        };
        if !ctx.is_available() {
            return;
        }
        let s0 = ctx.frame_slot();
        // Empty batch still acquires/submits a CB and advances slot
        let ok = unsafe { ctx.dispatch_optimizer_batch_async(&[]).is_ok() };
        if !ok {
            return;
        }
        let s1 = ctx.frame_slot();
        assert_ne!(s0, s1, "L-05 DoubleFrame must rotate slot after submit");
        let _ = unsafe { ctx.sync_frames() };
    }

    #[test]
    fn test_l06_rms_norm_matches_cpu() {
        let Ok(mut ctx) = AshContext::new() else {
            return;
        };
        if !ctx.is_available() {
            return;
        }
        let hidden = 64usize;
        let seq = 2usize;
        let eps = 1e-6f32;
        let mut x = vec![0.0f32; hidden * seq];
        let mut w = vec![1.0f32; hidden];
        for (i, xi) in x.iter_mut().enumerate() {
            *xi = (i as f32 * 0.01).sin();
        }
        for (i, wi) in w.iter_mut().enumerate() {
            *wi = 0.5 + (i as f32) * 0.001;
        }
        let mut y_gpu = vec![0.0f32; hidden * seq];
        let ok = unsafe {
            ctx.dispatch_rms_norm_sync(&x, &w, &mut y_gpu, hidden, seq, eps)
                .is_ok()
        };
        if !ok {
            return;
        }
        // CPU reference
        for s in 0..seq {
            let row = &x[s * hidden..(s + 1) * hidden];
            let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
            let scale = 1.0 / (mean_sq + eps).sqrt();
            for i in 0..hidden {
                let expect = row[i] * scale * w[i];
                let got = y_gpu[s * hidden + i];
                assert!(
                    (got - expect).abs() < 1e-3,
                    "rms mismatch pos={s} i={i}: gpu={got} cpu={expect}"
                );
            }
        }
        let _ = unsafe { ctx.sync() };
    }

    #[test]
    fn test_phase_b_gemv_shared_tile() {
        let Ok(mut ctx) = AshContext::new() else {
            return;
        };
        if !ctx.is_available() {
            return;
        }
        // Small but non-trivial: n_in=16 (2 u32/row), n_out=4
        let n_in = 16usize;
        let n_out = 4usize;
        let blocks = n_in / 8;
        let x: Vec<f32> = (0..n_in).map(|i| (i as f32 + 1.0) * 0.1).collect();
        // Each row: first nibble +1, rest 0 → y[r] = x[0] * scale[r]
        let mut packed = vec![0u32; n_out * blocks];
        for r in 0..n_out {
            packed[r * blocks] = 0x1; // only weight 0 = +1
        }
        let scales = vec![2.0f32; n_out];
        let mut y = vec![0.0f32; n_out];
        let ok = unsafe {
            ctx.dispatch_gemv_host_sync(&x, &packed, &scales, &mut y, n_in, n_out, false)
                .is_ok()
        };
        if !ok {
            return;
        }
        let expect = x[0] * 2.0;
        for (i, &yi) in y.iter().enumerate() {
            assert!(
                (yi - expect).abs() < 1e-3,
                "gemv row {i}: got {yi} expect {expect}"
            );
        }
        let _ = unsafe { ctx.sync() };
    }

    #[test]
    fn test_stream_f_qkv_one_cb() {
        let Ok(mut ctx) = AshContext::new() else {
            return;
        };
        if !ctx.is_available() {
            return;
        }
        let n_in = 16usize;
        let n_q = 8usize;
        let n_kv = 4usize;
        let blocks = n_in / 8;
        let x: Vec<f32> = (0..n_in).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let pack_rows = |n_out: usize, scale_nibble: u32| {
            let mut packed = vec![0u32; n_out * blocks];
            for r in 0..n_out {
                packed[r * blocks] = scale_nibble;
            }
            packed
        };
        // Q: weight0=+1 scale 1 → y=x[0]; K: +1 scale 2; V: +1 scale 3
        let q_p = pack_rows(n_q, 0x1);
        let k_p = pack_rows(n_kv, 0x1);
        let v_p = pack_rows(n_kv, 0x1);
        let q_sc = vec![1.0f32; n_q];
        let k_sc = vec![2.0f32; n_kv];
        let v_sc = vec![3.0f32; n_kv];
        let mut q_y = vec![0.0f32; n_q];
        let mut k_y = vec![0.0f32; n_kv];
        let mut v_y = vec![0.0f32; n_kv];
        let ok = unsafe {
            ctx.dispatch_gemv_qkv_host_sync(
                &x, &q_p, &k_p, &v_p, &q_sc, &k_sc, &v_sc, &mut q_y, &mut k_y, &mut v_y, n_in, n_q,
                n_kv, true, true, true,
            )
            .is_ok()
        };
        if !ok {
            return;
        }
        let eq = x[0];
        for (i, &yi) in q_y.iter().enumerate() {
            assert!((yi - eq).abs() < 1e-3, "Q[{i}]={yi} expect {eq}");
        }
        for (i, &yi) in k_y.iter().enumerate() {
            assert!((yi - eq * 2.0).abs() < 1e-3, "K[{i}]={yi}");
        }
        for (i, &yi) in v_y.iter().enumerate() {
            assert!((yi - eq * 3.0).abs() < 1e-3, "V[{i}]={yi}");
        }
        let _ = unsafe { ctx.sync() };
    }

    #[test]
    fn test_l06_mha_identity_like() {
        let Ok(mut ctx) = AshContext::new() else {
            return;
        };
        if !ctx.is_available() {
            return;
        }
        // seq=1 → causal attention is identity on V for matching Q=K directions
        let seq = 1usize;
        let n_head = 2usize;
        let n_kv = 2usize;
        let head_dim = 8usize;
        let q_n = seq * n_head * head_dim;
        let kv_n = seq * n_kv * head_dim;
        let mut q = vec![0.0f32; q_n];
        let mut k = vec![0.0f32; kv_n];
        let mut v = vec![0.0f32; kv_n];
        for i in 0..q_n {
            q[i] = 1.0;
            if i < kv_n {
                k[i] = 1.0;
                v[i] = (i % head_dim) as f32;
            }
        }
        let mut out = vec![0.0f32; q_n];
        let ok = unsafe {
            ctx.dispatch_mha_sync(&q, &k, &v, &mut out, seq, n_head, n_kv, head_dim)
                .is_ok()
        };
        if !ok {
            return;
        }
        // With seq=1, softmax is 1.0 → out == v per head (n_head == n_kv)
        for i in 0..q_n {
            assert!(
                (out[i] - v[i]).abs() < 1e-3,
                "mha seq1 identity fail i={i}: {} vs {}",
                out[i],
                v[i]
            );
        }
        let _ = unsafe { ctx.sync() };
    }
}
