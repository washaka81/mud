use std::collections::{HashMap, HashSet};
use std::sync::Arc;
pub mod vulkan_backend;
use parking_lot::Mutex;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateInfo, InstanceCreateFlags};
use vulkano::memory::allocator::{StandardMemoryAllocator, AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, layout::PipelineDescriptorSetLayoutCreateInfo};
use vulkano::VulkanLibrary;
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::sync::{self, GpuFuture};

pub struct VulkanContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub buffer_cache: Mutex<HashMap<String, Subbuffer<[u32]>>>,
    pub buffer_init: Mutex<HashSet<String>>,
    pub buffer_x_cache: Mutex<HashMap<String, Subbuffer<[f32]>>>,
    pub buffer_y_cache: Mutex<HashMap<String, Subbuffer<[f32]>>>,
    pub pipeline: Arc<ComputePipeline>,
    pub available: bool,
}

impl VulkanContext {
    pub fn is_available(&self) -> bool { self.available }

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
            .ok_or_else(|| anyhow::anyhow!("No se encontró ningún dispositivo Vulkan compatible"))?;

        let dev_props = physical_device.properties();
        println!("  🎮 GPU Detectada: {} ({:?})", dev_props.device_name, dev_props.device_type);

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_i, q)| q.queue_flags.contains(QueueFlags::COMPUTE))
            .ok_or_else(|| anyhow::anyhow!("No se encontró cola de COMPUTE"))? as u32;

        let (device, mut queues) = Device::new( physical_device,
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
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(device.clone(), Default::default()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(device.clone(), Default::default()));

        let shader = cs::load(device.clone())?;
        let entry_point = shader.entry_point("main").unwrap();
        let pipeline = ComputePipeline::new(
            device.clone(),
            None,
            vulkano::pipeline::compute::ComputePipelineCreateInfo::stage_layout(
                vulkano::pipeline::PipelineShaderStageCreateInfo::new(entry_point.clone()),
                PipelineLayout::new(
                    device.clone(),
                    PipelineDescriptorSetLayoutCreateInfo::from_stages([&vulkano::pipeline::PipelineShaderStageCreateInfo::new(entry_point.clone())])
                        .into_pipeline_layout_create_info(device.clone())?,
                )?,
            ),
        )?;

        Ok(Self { 
            device, queue, memory_allocator,
            command_buffer_allocator, descriptor_set_allocator,
            buffer_cache: Mutex::new(HashMap::new()),
            buffer_init: Mutex::new(HashSet::new()),
            buffer_x_cache: Mutex::new(HashMap::new()),
            buffer_y_cache: Mutex::new(HashMap::new()),
            pipeline,
            available: true,
        })
    }

    pub unsafe fn run_ternary_gemm_cached(
        &self,
        key: &str,
        batch_size: usize,
        n_in: usize,
        n_out: usize,
        x: &[f32],
        packed_w: *const u32,
        scale: f32,
        y: &mut [f32],
    ) -> anyhow::Result<()> {
        let buffer_x = {
            let mut cache = self.buffer_x_cache.lock();
            let total_x = batch_size * n_in;
            cache.entry(key.to_string()).or_insert_with(|| {
                Buffer::new_slice::<f32>(
                    self.memory_allocator.clone(),
                    BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
                    AllocationCreateInfo { 
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, 
                        ..Default::default() 
                    },
                    total_x as u64,
                ).unwrap()
            }).clone()
        };

        {
            let mut write_guard = buffer_x.write().unwrap();
            write_guard[..x.len()].copy_from_slice(x);
        }

        let buffer_w = {
            let mut cache = self.buffer_cache.lock();
            let w_len = (n_in / 16) * n_out;
            cache.entry(key.to_string()).or_insert_with(|| {
                let buf = Buffer::new_slice::<u32>(
                    self.memory_allocator.clone(),
                    BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
                    AllocationCreateInfo { 
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, 
                        ..Default::default() 
                    },
                    w_len as u64,
                ).unwrap();
                let weights_slice = unsafe { std::slice::from_raw_parts(packed_w, w_len) };
                buf.write().unwrap()[..w_len].copy_from_slice(weights_slice);
                buf
            }).clone()
        };

        let buffer_y = {
            let mut cache = self.buffer_y_cache.lock();
            let total_y = batch_size * n_out;
            cache.entry(key.to_string()).or_insert_with(|| {
                Buffer::new_slice::<f32>(
                    self.memory_allocator.clone(),
                    BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
                    AllocationCreateInfo { 
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_RANDOM_ACCESS, 
                        ..Default::default() 
                    },
                    total_y as u64,
                ).unwrap()
            }).clone()
        };

        let layout = self.pipeline.layout().set_layouts().first().unwrap();
        let set = PersistentDescriptorSet::new(
            &*self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, buffer_x.clone()),
                WriteDescriptorSet::buffer(1, buffer_w.clone()),
                WriteDescriptorSet::buffer(2, buffer_y.clone()),
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
            .bind_descriptor_sets(PipelineBindPoint::Compute, self.pipeline.layout().clone(), 0, set)?
            .push_constants(self.pipeline.layout().clone(), 0, cs::PushConstants { 
                n_in: n_in as u32, n_out: n_out as u32, scale, batch_size: batch_size as u32,
            })?
            .dispatch([n_out as u32, batch_size as u32, 1])?; 

        let command_buffer = builder.build()?;
        sync::now(self.device.clone()).then_execute(self.queue.clone(), command_buffer)?.then_signal_fence_and_flush()?.wait(None)?;

        {
            let read_guard = buffer_y.read().unwrap();
            y[..n_out * batch_size].copy_from_slice(&read_guard[..n_out * batch_size]);
        }
        Ok(())
    }

    pub unsafe fn run_ternary_gemv_cached(
        &self,
        key: &str,
        n_in: usize,
        n_out: usize,
        x: &[f32],
        packed_w: *const u32,
        scale: f32,
        y: &mut [f32],
    ) -> anyhow::Result<()> {
        unsafe { self.run_ternary_gemm_cached(key, 1, n_in, n_out, x, packed_w, scale, y) }
    }
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/ternary_gemv_igpu.comp",
        vulkan_version: "1.1",
    }
}
