use std::collections::{HashMap, HashSet};
use std::sync::Arc;
pub mod vulkan_backend;
use parking_lot::Mutex;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::{
    layout::PipelineDescriptorSetLayoutCreateInfo, ComputePipeline, Pipeline, PipelineBindPoint,
    PipelineLayout,
};
use vulkano::sync::{self, GpuFuture};
use vulkano::VulkanLibrary;

pub struct VulkanContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub buffer_cache: Mutex<HashMap<String, Subbuffer<[u32]>>>,
    pub scales_cache: Mutex<HashMap<String, Subbuffer<[f32]>>>,
    pub buffer_init: Mutex<HashSet<String>>,
    pub pipeline: Arc<ComputePipeline>,
    pub silu_pipeline: Arc<ComputePipeline>,
    pub heartbeat_pipeline: Arc<ComputePipeline>,
    pub backward_pipeline: Arc<ComputePipeline>,
    pub optimizer_pipeline: Arc<ComputePipeline>,
    pub ns_step1_pipeline: Arc<ComputePipeline>,
    pub ns_step2_pipeline: Arc<ComputePipeline>,
    pub telemetry_pipeline: Arc<ComputePipeline>,
    pub heartbeat_buffer: Subbuffer<[f32]>,
    pub available: bool,
}

impl VulkanContext {
    pub fn is_available(&self) -> bool {
        self.available
    }

    pub fn new() -> anyhow::Result<Self> {
        let use_vlk = std::env::var("MUD_USE_VULKAN").unwrap_or("1".to_string());
        if use_vlk == "0" || use_vlk.to_lowercase() == "false" {
            return Err(anyhow::anyhow!("Vulkan desactivado por MUD_USE_VULKAN"));
        }

        let library = VulkanLibrary::new()?;
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..InstanceCreateInfo::default()
            },
        )?;

        let physical_device = instance
            .enumerate_physical_devices()?
            .min_by_key(|p| {
                match p.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0, // Prefer Discrete if available
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    _ => 4,
                }
            })
            .ok_or_else(|| {
                anyhow::anyhow!("No se encontró ningún dispositivo Vulkan compatible")
            })?;

        // let dev_props = physical_device.properties();
        // println!("  🎮 GPU Detectada: {} ({:?})", dev_props.device_name, dev_props.device_type);

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_i, q)| q.queue_flags.contains(QueueFlags::COMPUTE))
            .ok_or_else(|| anyhow::anyhow!("No se encontró cola de COMPUTE"))?
            as u32;

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..QueueCreateInfo::default()
                }],
                enabled_extensions: DeviceExtensions {
                    khr_storage_buffer_storage_class: true,
                    ..DeviceExtensions::empty()
                },
                enabled_features: Features {
                    shader_subgroup_extended_types: true,
                    ..Features::empty()
                },
                ..DeviceCreateInfo::default()
            },
        )?;

        let queue = queues.next().unwrap();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let shader = cs::load(device.clone())?;
        let entry_point = shader.entry_point("main").unwrap();
        let pipeline = ComputePipeline::new(
            device.clone(),
            None,
            vulkano::pipeline::compute::ComputePipelineCreateInfo::stage_layout(
                vulkano::pipeline::PipelineShaderStageCreateInfo::new(entry_point.clone()),
                PipelineLayout::new(
                    device.clone(),
                    PipelineDescriptorSetLayoutCreateInfo::from_stages([
                        &vulkano::pipeline::PipelineShaderStageCreateInfo::new(entry_point.clone()),
                    ])
                    .into_pipeline_layout_create_info(device.clone())?,
                )?,
            ),
        )?;

        let silu_shader = silu_cs::load(device.clone())?;
        let silu_entry_point = silu_shader.entry_point("main").unwrap();
        let silu_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            vulkano::pipeline::compute::ComputePipelineCreateInfo::stage_layout(
                vulkano::pipeline::PipelineShaderStageCreateInfo::new(silu_entry_point.clone()),
                PipelineLayout::new(
                    device.clone(),
                    PipelineDescriptorSetLayoutCreateInfo::from_stages([
                        &vulkano::pipeline::PipelineShaderStageCreateInfo::new(
                            silu_entry_point.clone(),
                        ),
                    ])
                    .into_pipeline_layout_create_info(device.clone())?,
                )?,
            ),
        )?;

        let heartbeat_shader = heartbeat_cs::load(device.clone())?;
        let heartbeat_entry_point = heartbeat_shader.entry_point("main").unwrap();
        let heartbeat_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            vulkano::pipeline::compute::ComputePipelineCreateInfo::stage_layout(
                vulkano::pipeline::PipelineShaderStageCreateInfo::new(
                    heartbeat_entry_point.clone(),
                ),
                PipelineLayout::new(
                    device.clone(),
                    PipelineDescriptorSetLayoutCreateInfo::from_stages([
                        &vulkano::pipeline::PipelineShaderStageCreateInfo::new(
                            heartbeat_entry_point.clone(),
                        ),
                    ])
                    .into_pipeline_layout_create_info(device.clone())?,
                )?,
            ),
        )?;

        let backward_shader = backward_cs::load(device.clone())?;
        let backward_entry_point = backward_shader.entry_point("main").unwrap();
        let backward_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            vulkano::pipeline::compute::ComputePipelineCreateInfo::stage_layout(
                vulkano::pipeline::PipelineShaderStageCreateInfo::new(
                    backward_entry_point.clone(),
                ),
                PipelineLayout::new(
                    device.clone(),
                    PipelineDescriptorSetLayoutCreateInfo::from_stages([
                        &vulkano::pipeline::PipelineShaderStageCreateInfo::new(
                            backward_entry_point.clone(),
                        ),
                    ])
                    .into_pipeline_layout_create_info(device.clone())?,
                )?,
            ),
        )?;

        let optimizer_shader = optimizer_cs::load(device.clone())?;
        let optimizer_entry_point = optimizer_shader.entry_point("main").unwrap();
        let optimizer_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            vulkano::pipeline::compute::ComputePipelineCreateInfo::stage_layout(
                vulkano::pipeline::PipelineShaderStageCreateInfo::new(
                    optimizer_entry_point.clone(),
                ),
                PipelineLayout::new(
                    device.clone(),
                    PipelineDescriptorSetLayoutCreateInfo::from_stages([
                        &vulkano::pipeline::PipelineShaderStageCreateInfo::new(
                            optimizer_entry_point.clone(),
                        ),
                    ])
                    .into_pipeline_layout_create_info(device.clone())?,
                )?,
            ),
        )?;

        let ns_step1_shader = newton_schulz_step1_cs::load(device.clone())?;
        let ns_step1_entry_point = ns_step1_shader.entry_point("main").unwrap();
        let ns_step1_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            vulkano::pipeline::compute::ComputePipelineCreateInfo::stage_layout(
                vulkano::pipeline::PipelineShaderStageCreateInfo::new(ns_step1_entry_point.clone()),
                PipelineLayout::new(
                    device.clone(),
                    PipelineDescriptorSetLayoutCreateInfo::from_stages([
                        &vulkano::pipeline::PipelineShaderStageCreateInfo::new(
                            ns_step1_entry_point.clone(),
                        ),
                    ])
                    .into_pipeline_layout_create_info(device.clone())?,
                )?,
            ),
        )?;

        let ns_step2_shader = newton_schulz_step2_cs::load(device.clone())?;
        let ns_step2_entry_point = ns_step2_shader.entry_point("main").unwrap();
        let ns_step2_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            vulkano::pipeline::compute::ComputePipelineCreateInfo::stage_layout(
                vulkano::pipeline::PipelineShaderStageCreateInfo::new(ns_step2_entry_point.clone()),
                PipelineLayout::new(
                    device.clone(),
                    PipelineDescriptorSetLayoutCreateInfo::from_stages([
                        &vulkano::pipeline::PipelineShaderStageCreateInfo::new(
                            ns_step2_entry_point.clone(),
                        ),
                    ])
                    .into_pipeline_layout_create_info(device.clone())?,
                )?,
            ),
        )?;

        let telemetry_shader = telemetry_cs::load(device.clone())?;
        let telemetry_entry_point = telemetry_shader.entry_point("main").unwrap();
        let telemetry_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            vulkano::pipeline::compute::ComputePipelineCreateInfo::stage_layout(
                vulkano::pipeline::PipelineShaderStageCreateInfo::new(telemetry_entry_point.clone()),
                PipelineLayout::new(
                    device.clone(),
                    PipelineDescriptorSetLayoutCreateInfo::from_stages([
                        &vulkano::pipeline::PipelineShaderStageCreateInfo::new(
                            telemetry_entry_point.clone(),
                        ),
                    ])
                    .into_pipeline_layout_create_info(device.clone())?,
                )?,
            ),
        )?;

        let heartbeat_buffer = Buffer::new_slice::<f32>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            1,
        )?;
        heartbeat_buffer.write()?[0] = 0.0;

        Ok(Self {
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
            buffer_cache: Mutex::new(HashMap::new()),
            scales_cache: Mutex::new(HashMap::new()),
            buffer_init: Mutex::new(HashSet::new()),
            pipeline,
            silu_pipeline,
            heartbeat_pipeline,
            backward_pipeline,
            optimizer_pipeline,
            ns_step1_pipeline,
            ns_step2_pipeline,
            telemetry_pipeline,
            heartbeat_buffer,
            available: true,
        })
    }

    /// # Safety
    ///
    /// This function is unsafe because it performs a raw Vulkan dispatch.
    /// It assumes the heartbeat pipeline and buffer are correctly initialized.
    pub unsafe fn pulse_heartbeat(&self) {
        let layout = self
            .heartbeat_pipeline
            .layout()
            .set_layouts()
            .first()
            .unwrap();
        let set = PersistentDescriptorSet::new(
            &*self.descriptor_set_allocator,
            layout.clone(),
            [WriteDescriptorSet::buffer(0, self.heartbeat_buffer.clone())],
            [],
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            &*self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.heartbeat_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.heartbeat_pipeline.layout().clone(),
                0,
                set,
            )
            .unwrap()
            .dispatch([1, 1, 1])
            .unwrap();

        let command_buffer = builder.build().unwrap();
        let _ = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush();
    }

    /// RRM: Asynchronous Imagination
    /// Dispatches a speculative compute shader on Vulkan and returns the future
    /// so the CPU can perform LDT evaluation in parallel.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it performs a raw Vulkan dispatch
    /// and returns a GpuFuture that the caller must manage.
    pub unsafe fn dispatch_imagination_async(&self) -> Box<dyn GpuFuture> {
        let layout = self
            .heartbeat_pipeline
            .layout()
            .set_layouts()
            .first()
            .unwrap();
        let set = PersistentDescriptorSet::new(
            &*self.descriptor_set_allocator,
            layout.clone(),
            [WriteDescriptorSet::buffer(0, self.heartbeat_buffer.clone())],
            [],
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            &*self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.heartbeat_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.heartbeat_pipeline.layout().clone(),
                0,
                set,
            )
            .unwrap()
            .dispatch([64, 1, 1]) // Larger dispatch for speculative work
            .unwrap();

        let command_buffer = builder.build().unwrap();
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
            
        Box::new(future)
    }

    pub fn allocate_zero_copy_buffer(&self, len: usize) -> Subbuffer<[f32]> {
        Buffer::new_slice::<f32>(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            len as u64,
        )
        .unwrap()
    }

    pub fn allocate_zero_copy_buffer_u32(&self, len: usize) -> Subbuffer<[u32]> {
        Buffer::new_slice::<u32>(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            len as u64,
        )
        .unwrap()
    }

    /// # Safety
    ///
    /// This function is unsafe because it dereferences the `scales` raw pointer.
    /// The caller must ensure that `scales` points to at least `n_out` valid `f32` elements if not null.
    pub unsafe fn get_or_create_scales_buffer(
        &self,
        key: &str,
        n_out: usize,
        scales: *const f32,
    ) -> Subbuffer<[f32]> {
        let mut cache = self.scales_cache.lock();
        let scales_key = format!("{}_scales", key);
        cache
            .entry(scales_key)
            .or_insert_with(|| {
                let buf = Buffer::new_slice::<f32>(
                    self.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    n_out as u64,
                )
                .unwrap();
                let mut write_guard = buf.write().unwrap();
                if scales.is_null() {
                    write_guard[..n_out].fill(1.0);
                } else {
                    let scales_slice = unsafe { std::slice::from_raw_parts(scales, n_out) };
                    write_guard[..n_out].copy_from_slice(scales_slice);
                }
                drop(write_guard);
                buf
            })
            .clone()
    }

    /// # Safety
    ///
    /// This function is unsafe because it dereferences `packed_w` and `scales` raw pointers.
    /// The caller must ensure they point to valid memory according to the specified dimensions.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_ternary_gemm_cached(
        &self,
        key: &str,
        batch_size: usize,
        n_in: usize,
        n_out: usize,
        buffer_x: &Subbuffer<[f32]>,
        packed_w: *const u32,
        scales: *const f32,
        buffer_y: &Subbuffer<[f32]>,
    ) -> anyhow::Result<()> {
        let buffer_scales = self.get_or_create_scales_buffer(key, n_out, scales);

        let buffer_w = {
            let mut cache = self.buffer_cache.lock();
            let w_len = (n_in / 16) * n_out;
            cache
                .entry(key.to_string())
                .or_insert_with(|| {
                    let buf = Buffer::new_slice::<u32>(
                        self.memory_allocator.clone(),
                        BufferCreateInfo {
                            usage: BufferUsage::STORAGE_BUFFER,
                            ..Default::default()
                        },
                        AllocationCreateInfo {
                            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                            ..Default::default()
                        },
                        w_len as u64,
                    )
                    .unwrap();
                    let weights_slice = unsafe { std::slice::from_raw_parts(packed_w, w_len) };
                    buf.write().unwrap()[..w_len].copy_from_slice(weights_slice);
                    buf
                })
                .clone()
        };

        let layout = self.pipeline.layout().set_layouts().first().unwrap();
        let set = PersistentDescriptorSet::new(
            &*self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, buffer_x.clone()),
                WriteDescriptorSet::buffer(1, buffer_w.clone()),
                WriteDescriptorSet::buffer(2, buffer_y.clone()),
                WriteDescriptorSet::buffer(3, buffer_scales.clone()),
            ],
            [],
        )?;

        let mut builder = AutoCommandBufferBuilder::primary(
            &*self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        builder
            .bind_pipeline_compute(self.pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                set,
            )?
            .push_constants(
                self.pipeline.layout().clone(),
                0,
                cs::PushConstants {
                    n_in: n_in as u32,
                    n_out: n_out as u32,
                    batch_size: batch_size as u32,
                    inv_q_scale: 1.0,
                    do_norm: 0,
                    single_scale: 0,
                    scale: 1.0,
                },
            )?
            .dispatch([n_out as u32, batch_size as u32, 1])?;

        let command_buffer = builder.build()?;
        sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?
            .wait(None)?;

        Ok(())
    }

    /// # Safety
    ///
    /// This function is unsafe because it dereferences `packed_w` and `scales` raw pointers.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_ternary_gemv_cached(
        &self,
        key: &str,
        n_in: usize,
        n_out: usize,
        buffer_x: &Subbuffer<[f32]>,
        packed_w: *const u32,
        scales: *const f32,
        buffer_y: &Subbuffer<[f32]>,
    ) -> anyhow::Result<()> {
        unsafe {
            self.run_ternary_gemm_cached(key, 1, n_in, n_out, buffer_x, packed_w, scales, buffer_y)
        }
    }

    /// # Safety
    ///
    /// This function is unsafe because it dereferences `packed_w` and `scales` raw pointers.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_ternary_gemm_cached_async(
        &self,
        key: &str,
        batch_size: usize,
        n_in: usize,
        n_out: usize,
        buffer_x: &Subbuffer<[f32]>,
        packed_w: *const u32,
        scales: *const f32,
        buffer_y: &Subbuffer<[f32]>,
    ) -> anyhow::Result<()> {
        let buffer_scales = self.get_or_create_scales_buffer(key, n_out, scales);
        let buffer_w = {
            let mut cache = self.buffer_cache.lock();
            let w_len = (n_in / 16) * n_out;
            cache
                .entry(key.to_string())
                .or_insert_with(|| {
                    let buf = Buffer::new_slice::<u32>(
                        self.memory_allocator.clone(),
                        BufferCreateInfo {
                            usage: BufferUsage::STORAGE_BUFFER,
                            ..Default::default()
                        },
                        AllocationCreateInfo {
                            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                            ..Default::default()
                        },
                        w_len as u64,
                    )
                    .unwrap();
                    let weights_slice = unsafe { std::slice::from_raw_parts(packed_w, w_len) };
                    buf.write().unwrap()[..w_len].copy_from_slice(weights_slice);
                    buf
                })
                .clone()
        };

        let layout = self.pipeline.layout().set_layouts().first().unwrap();
        let set = PersistentDescriptorSet::new(
            &*self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, buffer_x.clone()),
                WriteDescriptorSet::buffer(1, buffer_w.clone()),
                WriteDescriptorSet::buffer(2, buffer_y.clone()),
                WriteDescriptorSet::buffer(3, buffer_scales.clone()),
            ],
            [],
        )?;

        let mut builder = AutoCommandBufferBuilder::primary(
            &*self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        builder
            .bind_pipeline_compute(self.pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                set,
            )?
            .push_constants(
                self.pipeline.layout().clone(),
                0,
                cs::PushConstants {
                    n_in: n_in as u32,
                    n_out: n_out as u32,
                    batch_size: batch_size as u32,
                    inv_q_scale: 1.0,
                    do_norm: 0,
                    single_scale: 0,
                    scale: 1.0,
                },
            )?
            .dispatch([n_out as u32, batch_size as u32, 1])?;

        let command_buffer = builder.build()?;
        let _future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?;

        Ok(())
    }

    /// # Safety
    ///
    /// This function is unsafe because it dereferences `packed_w` and `scales` raw pointers.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_ternary_gemv_cached_async(
        &self,
        key: &str,
        n_in: usize,
        n_out: usize,
        buffer_x: &Subbuffer<[f32]>,
        packed_w: *const u32,
        scales: *const f32,
        buffer_y: &Subbuffer<[f32]>,
    ) -> anyhow::Result<()> {
        unsafe {
            self.run_ternary_gemm_cached_async(
                key, 1, n_in, n_out, buffer_x, packed_w, scales, buffer_y,
            )
        }
    }

    /// # Safety
    ///
    /// This function is unsafe because it dereferences multiple weight and scale raw pointers.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_chained_ffn(
        &self,
        key_w1: &str,
        key_w2: &str,
        key_w3: &str,
        hidden: usize,
        ffn_hidden: usize,
        buffer_x: &Subbuffer<[f32]>,
        w1_packed: *const u32,
        w1_scales: *const f32,
        buffer_w1_out: &Subbuffer<[f32]>,
        w3_packed: *const u32,
        w3_scales: *const f32,
        buffer_w3_out: &Subbuffer<[f32]>,
        w2_packed: *const u32,
        w2_scales: *const f32,
        buffer_final_out: &Subbuffer<[f32]>,
    ) -> anyhow::Result<()> {
        let buffer_w1_scales = self.get_or_create_scales_buffer(key_w1, ffn_hidden, w1_scales);
        let buffer_w3_scales = self.get_or_create_scales_buffer(key_w3, ffn_hidden, w3_scales);
        let buffer_w2_scales = self.get_or_create_scales_buffer(key_w2, hidden, w2_scales);
        let buffer_w1 = {
            let mut cache = self.buffer_cache.lock();
            let w_len = (hidden / 16) * ffn_hidden;
            cache
                .entry(key_w1.to_string())
                .or_insert_with(|| {
                    let buf = Buffer::new_slice::<u32>(
                        self.memory_allocator.clone(),
                        BufferCreateInfo {
                            usage: BufferUsage::STORAGE_BUFFER,
                            ..Default::default()
                        },
                        AllocationCreateInfo {
                            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                            ..Default::default()
                        },
                        w_len as u64,
                    )
                    .unwrap();
                    let weights_slice = unsafe { std::slice::from_raw_parts(w1_packed, w_len) };
                    buf.write().unwrap()[..w_len].copy_from_slice(weights_slice);
                    buf
                })
                .clone()
        };

        let buffer_w3 = {
            let mut cache = self.buffer_cache.lock();
            let w_len = (hidden / 16) * ffn_hidden;
            cache
                .entry(key_w3.to_string())
                .or_insert_with(|| {
                    let buf = Buffer::new_slice::<u32>(
                        self.memory_allocator.clone(),
                        BufferCreateInfo {
                            usage: BufferUsage::STORAGE_BUFFER,
                            ..Default::default()
                        },
                        AllocationCreateInfo {
                            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                            ..Default::default()
                        },
                        w_len as u64,
                    )
                    .unwrap();
                    let weights_slice = unsafe { std::slice::from_raw_parts(w3_packed, w_len) };
                    buf.write().unwrap()[..w_len].copy_from_slice(weights_slice);
                    buf
                })
                .clone()
        };

        let buffer_w2 = {
            let mut cache = self.buffer_cache.lock();
            let w_len = (ffn_hidden / 16) * hidden;
            cache
                .entry(key_w2.to_string())
                .or_insert_with(|| {
                    let buf = Buffer::new_slice::<u32>(
                        self.memory_allocator.clone(),
                        BufferCreateInfo {
                            usage: BufferUsage::STORAGE_BUFFER,
                            ..Default::default()
                        },
                        AllocationCreateInfo {
                            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                            ..Default::default()
                        },
                        w_len as u64,
                    )
                    .unwrap();
                    let weights_slice = unsafe { std::slice::from_raw_parts(w2_packed, w_len) };
                    buf.write().unwrap()[..w_len].copy_from_slice(weights_slice);
                    buf
                })
                .clone()
        };

        let layout_gemv = self.pipeline.layout().set_layouts().first().unwrap();
        let set_w1 = PersistentDescriptorSet::new(
            &*self.descriptor_set_allocator,
            layout_gemv.clone(),
            [
                WriteDescriptorSet::buffer(0, buffer_x.clone()),
                WriteDescriptorSet::buffer(1, buffer_w1.clone()),
                WriteDescriptorSet::buffer(2, buffer_w1_out.clone()),
                WriteDescriptorSet::buffer(3, buffer_w1_scales.clone()),
            ],
            [],
        )?;

        let set_w3 = PersistentDescriptorSet::new(
            &*self.descriptor_set_allocator,
            layout_gemv.clone(),
            [
                WriteDescriptorSet::buffer(0, buffer_x.clone()),
                WriteDescriptorSet::buffer(1, buffer_w3.clone()),
                WriteDescriptorSet::buffer(2, buffer_w3_out.clone()),
                WriteDescriptorSet::buffer(3, buffer_w3_scales.clone()),
            ],
            [],
        )?;

        let layout_silu = self.silu_pipeline.layout().set_layouts().first().unwrap();
        let set_silu = PersistentDescriptorSet::new(
            &*self.descriptor_set_allocator,
            layout_silu.clone(),
            [
                WriteDescriptorSet::buffer(0, buffer_w1_out.clone()),
                WriteDescriptorSet::buffer(1, buffer_w3_out.clone()),
                WriteDescriptorSet::buffer(2, buffer_w1_out.clone()),
            ],
            [],
        )?;

        let set_w2 = PersistentDescriptorSet::new(
            &*self.descriptor_set_allocator,
            layout_gemv.clone(),
            [
                WriteDescriptorSet::buffer(0, buffer_w1_out.clone()),
                WriteDescriptorSet::buffer(1, buffer_w2.clone()),
                WriteDescriptorSet::buffer(2, buffer_final_out.clone()),
                WriteDescriptorSet::buffer(3, buffer_w2_scales.clone()),
            ],
            [],
        )?;

        let mut builder = AutoCommandBufferBuilder::primary(
            &*self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        builder
            .bind_pipeline_compute(self.pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                set_w1,
            )?
            .push_constants(
                self.pipeline.layout().clone(),
                0,
                cs::PushConstants {
                    n_in: hidden as u32,
                    n_out: ffn_hidden as u32,
                    batch_size: 1,
                    inv_q_scale: 1.0,
                    do_norm: 0,
                    single_scale: 0,
                    scale: 1.0,
                },
            )?
            .dispatch([ffn_hidden as u32, 1, 1])?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                set_w3,
            )?
            .push_constants(
                self.pipeline.layout().clone(),
                0,
                cs::PushConstants {
                    n_in: hidden as u32,
                    n_out: ffn_hidden as u32,
                    batch_size: 1,
                    inv_q_scale: 1.0,
                    do_norm: 0,
                    single_scale: 0,
                    scale: 1.0,
                },
            )?
            .dispatch([ffn_hidden as u32, 1, 1])?
            .bind_pipeline_compute(self.silu_pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.silu_pipeline.layout().clone(),
                0,
                set_silu,
            )?
            .push_constants(
                self.silu_pipeline.layout().clone(),
                0,
                silu_cs::PushConstants {
                    size: ffn_hidden as u32,
                    do_relu2: 0,
                },
            )?
            .dispatch([ffn_hidden.div_ceil(256) as u32, 1, 1])?
            .bind_pipeline_compute(self.pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                set_w2,
            )?
            .push_constants(
                self.pipeline.layout().clone(),
                0,
                cs::PushConstants {
                    n_in: ffn_hidden as u32,
                    n_out: hidden as u32,
                    batch_size: 1,
                    inv_q_scale: 1.0,
                    do_norm: 0,
                    single_scale: 0,
                    scale: 1.0,
                },
            )?
            .dispatch([hidden as u32, 1, 1])?;

        let command_buffer = builder.build()?;
        sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?
            .wait(None)?;

        Ok(())
    }

    /// # Safety
    ///
    /// This function is unsafe because it dereferences multiple weight and gradient raw pointers.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_qat_backward_async(
        &self,
        n_in: usize,
        n_out: usize,
        batch_size: usize,
        buffer_x: &Subbuffer<[f32]>,
        buffer_grad_y: &Subbuffer<[f32]>,
        buffer_grad_w: &Subbuffer<[f32]>,
    ) -> anyhow::Result<()> {
        let layout = self.backward_pipeline.layout().set_layouts().first().unwrap();
        let set = PersistentDescriptorSet::new(
            &*self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, buffer_x.clone()),
                WriteDescriptorSet::buffer(1, buffer_grad_y.clone()),
                WriteDescriptorSet::buffer(2, buffer_grad_w.clone()),
            ],
            [],
        )?;

        let mut builder = AutoCommandBufferBuilder::primary(
            &*self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        builder
            .bind_pipeline_compute(self.backward_pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backward_pipeline.layout().clone(),
                0,
                set,
            )?
            .push_constants(
                self.backward_pipeline.layout().clone(),
                0,
                backward_cs::PushConstants {
                    n_in: n_in as u32,
                    n_out: n_out as u32,
                    batch_size: batch_size as u32,
                },
            )?
            .dispatch([
                (n_in as u32).div_ceil(32),
                (n_out as u32).div_ceil(8),
                1,
            ])?;

        let command_buffer = builder.build()?;
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?;
        future.wait(None)?;

        Ok(())
    }

    /// # Safety
    ///
    /// This function is unsafe because it dereferences multiple shadow weight and scale raw pointers.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_qat_optimizer_async(
        &self,
        total_elements: usize,
        cols: usize,
        learning_rate: f32,
        weight_decay: f32,
        buffer_shadow_w: &Subbuffer<[f32]>,
        buffer_grad_w: &Subbuffer<[f32]>,
        buffer_scales: &Subbuffer<[f32]>,
        buffer_packed_w: &Subbuffer<[u32]>,
    ) -> anyhow::Result<()> {
        let layout = self.optimizer_pipeline.layout().set_layouts().first().unwrap();
        let set = PersistentDescriptorSet::new(
            &*self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, buffer_shadow_w.clone()),
                WriteDescriptorSet::buffer(1, buffer_grad_w.clone()),
                WriteDescriptorSet::buffer(2, buffer_scales.clone()),
                WriteDescriptorSet::buffer(3, buffer_packed_w.clone()),
            ],
            [],
        )?;

        let mut builder = AutoCommandBufferBuilder::primary(
            &*self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        builder
            .bind_pipeline_compute(self.optimizer_pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.optimizer_pipeline.layout().clone(),
                0,
                set,
            )?
            .push_constants(
                self.optimizer_pipeline.layout().clone(),
                0,
                optimizer_cs::PushConstants {
                    total_elements: total_elements as u32,
                    cols: cols as u32,
                    learning_rate,
                    weight_decay,
                },
            )?
            .dispatch([(total_elements as u32) / (cols as u32), 1, 1])?;

        let command_buffer = builder.build()?;
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?;
        future.wait(None)?;

        Ok(())
    }

    /// # Safety
    /// Caller must ensure that `updates` correctly maps to the underlying buffers and sizes.
    pub unsafe fn run_qat_optimizer_batch(
        &self,
        updates: &[(usize, usize, f32, f32, vulkano::buffer::Subbuffer<[f32]>, vulkano::buffer::Subbuffer<[f32]>, vulkano::buffer::Subbuffer<[f32]>, vulkano::buffer::Subbuffer<[u32]>)]
    ) -> anyhow::Result<()> {
        let mut builder = AutoCommandBufferBuilder::primary(
            &*self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        let layout = self.optimizer_pipeline.layout().set_layouts().first().unwrap();

        for (total_elements, cols, learning_rate, weight_decay, buffer_shadow_w, buffer_grad_w, buffer_scales, buffer_packed_w) in updates {
            let set = PersistentDescriptorSet::new(
                &*self.descriptor_set_allocator,
                layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, buffer_shadow_w.clone()),
                    WriteDescriptorSet::buffer(1, buffer_grad_w.clone()),
                    WriteDescriptorSet::buffer(2, buffer_scales.clone()),
                    WriteDescriptorSet::buffer(3, buffer_packed_w.clone()),
                ],
                [],
            )?;

            builder
                .bind_pipeline_compute(self.optimizer_pipeline.clone())?
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.optimizer_pipeline.layout().clone(),
                    0,
                    set,
                )?
                .push_constants(
                    self.optimizer_pipeline.layout().clone(),
                    0,
                    optimizer_cs::PushConstants {
                        total_elements: *total_elements as u32,
                        cols: *cols as u32,
                        learning_rate: *learning_rate,
                        weight_decay: *weight_decay,
                    },
                )?
                .dispatch([(*total_elements as u32) / (*cols as u32), 1, 1])?;
        }

        let command_buffer = builder.build()?;
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?;
        future.wait(None)?;

        Ok(())
    }


    /// # Safety
    ///
    /// This function is unsafe because it performs a raw Vulkan compute dispatch.
    /// The caller must ensure that the buffers have sufficient capacity for `rows * cols` elements.
    pub unsafe fn run_newton_schulz_async(
        &self,
        rows: usize,
        cols: usize,
        n_iters: usize,
        buffer_x: &Subbuffer<[f32]>,
        buffer_tmp: &Subbuffer<[f32]>,
        buffer_next_x: &Subbuffer<[f32]>,
    ) -> anyhow::Result<()> {
        let mut builder = AutoCommandBufferBuilder::primary(
            &*self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        let layout1 = self.ns_step1_pipeline.layout().set_layouts().first().unwrap();
        let set1 = PersistentDescriptorSet::new(
            &*self.descriptor_set_allocator,
            layout1.clone(),
            [
                WriteDescriptorSet::buffer(0, buffer_x.clone()),
                WriteDescriptorSet::buffer(1, buffer_tmp.clone()),
            ],
            [],
        )?;

        let layout2 = self.ns_step2_pipeline.layout().set_layouts().first().unwrap();
        let set2 = PersistentDescriptorSet::new(
            &*self.descriptor_set_allocator,
            layout2.clone(),
            [
                WriteDescriptorSet::buffer(0, buffer_x.clone()),
                WriteDescriptorSet::buffer(1, buffer_tmp.clone()),
                WriteDescriptorSet::buffer(2, buffer_next_x.clone()),
            ],
            [],
        )?;

        for _ in 0..n_iters {
            builder
                .bind_pipeline_compute(self.ns_step1_pipeline.clone())?
                .bind_descriptor_sets(PipelineBindPoint::Compute, self.ns_step1_pipeline.layout().clone(), 0, set1.clone())?
                .push_constants(self.ns_step1_pipeline.layout().clone(), 0, newton_schulz_step1_cs::PushConstants { rows: rows as u32, cols: cols as u32 })?
                .dispatch([(cols as u32).div_ceil(16), (cols as u32).div_ceil(16), 1])?;

            builder
                .bind_pipeline_compute(self.ns_step2_pipeline.clone())?
                .bind_descriptor_sets(PipelineBindPoint::Compute, self.ns_step2_pipeline.layout().clone(), 0, set2.clone())?
                .push_constants(self.ns_step2_pipeline.layout().clone(), 0, newton_schulz_step2_cs::PushConstants { rows: rows as u32, cols: cols as u32 })?
                .dispatch([(cols as u32).div_ceil(16), (rows as u32).div_ceil(16), 1])?;

            builder.copy_buffer(vulkano::command_buffer::CopyBufferInfo::buffers(
                buffer_next_x.clone(),
                buffer_x.clone(),
            ))?;
        }

        let command_buffer = builder.build()?;
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?;
        // Wait asynchronously instead of blocking strictly? For now blocking is fine to ensure completion.
        future.wait(None)?;

        Ok(())
    }

    /// # Safety
    ///
    /// This function is unsafe because it performs a raw Vulkan compute dispatch.
    pub unsafe fn run_telemetry_async(
        &self,
        size: usize,
        buffer_in: &Subbuffer<[f32]>,
        buffer_out: &Subbuffer<[f32]>,
    ) -> anyhow::Result<()> {
        let mut builder = AutoCommandBufferBuilder::primary(
            &*self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        let layout = self.telemetry_pipeline.layout().set_layouts().first().unwrap();
        let set = PersistentDescriptorSet::new(
            &*self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, buffer_in.clone()),
                WriteDescriptorSet::buffer(1, buffer_out.clone()),
            ],
            [],
        )?;

        builder
            .bind_pipeline_compute(self.telemetry_pipeline.clone())?
            .bind_descriptor_sets(PipelineBindPoint::Compute, self.telemetry_pipeline.layout().clone(), 0, set.clone())?
            .push_constants(self.telemetry_pipeline.layout().clone(), 0, telemetry_cs::PushConstants { size: size as u32 })?
            .dispatch([1, 1, 1])?;

        let command_buffer = builder.build()?;
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?;

        future.wait(None)?;

        Ok(())
    }
}
mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/ternary_gemv_unified.comp",
        vulkan_version: "1.1",
    }
}

mod heartbeat_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/heartbeat.comp",
        vulkan_version: "1.1",
    }
}

mod silu_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/silu_gate.comp",
        vulkan_version: "1.1",
    }
}

pub mod backward_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/ternary_backward.comp",
        vulkan_version: "1.1",
    }
}

pub mod optimizer_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/shadow_optimizer.comp",
        vulkan_version: "1.1",
    }
}

pub mod newton_schulz_step1_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/newton_schulz_step1.comp",
        vulkan_version: "1.1",
    }
}

pub mod newton_schulz_step2_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/newton_schulz_step2.comp",
        vulkan_version: "1.1",
    }
}

pub mod telemetry_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/tensor_thermodynamics.comp",
        vulkan_version: "1.1",
    }
}

pub mod dspark_drafter_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/dspark_drafter.comp",
        vulkan_version: "1.1",
    }
}

/// DSpark Asynchronous Ring Buffer (Priority 50)
/// Decouples the Vulkan Drafter from the CPU Verifier by running the Drafter
/// in an independent background thread, pushing proposals to an MPSC Ring Buffer.
pub struct DSparkRingBuffer {
    pub receiver: std::sync::mpsc::Receiver<Vec<u32>>,
}

impl DSparkRingBuffer {
    pub fn new(_device: std::sync::Arc<vulkano::device::Device>, _queue: std::sync::Arc<vulkano::device::Queue>) -> Self {
        let (_tx, rx) = std::sync::mpsc::channel();
        
        // Asynchronous Vulkan submission loop
        std::thread::spawn(move || {
            // Placeholder: this loop will continuously submit dspark_drafter.comp
            // and push the resulting candidates (e.g. K=3 tokens) to tx.
            // Currently yields to prevent blocking since the shader is a skeleton.
            loop {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        });

        Self { receiver: rx }
    }
}
