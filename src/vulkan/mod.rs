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

/// Manages the Vulkan compute environment for iGPU offloading.
/// Specifically tuned for Intel Iris Xe (ADL GT2) hardware.
pub struct VulkanContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    /// Persistent cache for weight buffers to avoid re-allocation.
    pub buffer_cache: Mutex<HashMap<String, Arc<Subbuffer<[u32]>>>>,
    /// Tracks which weight buffers have been initialized (data written) to skip re-upload.
    pub buffer_init: Mutex<HashSet<String>>,
    /// Persistent cache for input buffers to avoid re-allocation.
    pub buffer_x_cache: Mutex<HashMap<String, Arc<Subbuffer<[f32]>>>>,
    /// Persistent cache for output buffers to avoid re-allocation.
    pub buffer_y_cache: Mutex<HashMap<String, Arc<Subbuffer<[f32]>>>>,
    /// Cached compute pipeline.
    pub pipeline: Arc<ComputePipeline>,
    /// Whether Vulkan was successfully initialized.
    pub available: bool,
}

impl VulkanContext {
    pub fn is_available(&self) -> bool { self.available }

    /// Initializes a new Vulkan context.
    pub fn new() -> anyhow::Result<Self> {
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
                    PhysicalDeviceType::IntegratedGpu => 0,
                    PhysicalDeviceType::DiscreteGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    _ => 4,
                }
            })
            .ok_or_else(|| anyhow::anyhow!("No se encontró ningún dispositivo Vulkan compatible"))?;

        println!("[Vulkan] Usando dispositivo: {} (tipo: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type);

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

    /// Executes a ternary GEMV on the iGPU with buffer caching.
    pub fn run_ternary_gemv_cached(
        &self,
        key: &str,
        n_in: usize,
        n_out: usize,
        x: &[f32],
        packed_w: *const u32,
        scale: f32,
        y: &mut [f32],
    ) -> anyhow::Result<()> {
        // Use cached input buffer if available
        let buffer_x = {
            let mut cache = self.buffer_x_cache.lock();
            let mut recreate = true;
            if let Some(buf) = cache.get(key) {
                if buf.len() == n_in as u64 {
                    recreate = false;
                }
            }
            if recreate {
                let buf = Buffer::new_slice::<f32>(
                    self.memory_allocator.clone(),
                    BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
                    AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, ..Default::default() },
                    n_in as u64,
                )?;
                let arc_buf = Arc::new(buf);
                cache.insert(key.to_string(), arc_buf.clone());
                arc_buf
            } else {
                cache.get(key).unwrap().clone()
            }
        };

        // Persistent host write directly into mapped memory buffer
        {
            let mut write_guard = buffer_x.write()?;
            let copy_len = x.len().min(write_guard.len());
            write_guard[..copy_len].copy_from_slice(&x[..copy_len]);
        }

        // Use cached weight buffer if available (zero-copy: skip re-upload on cache hit)
        let buffer_w: Arc<Subbuffer<[u32]>> = {
            let mut cache = self.buffer_cache.lock();
            let mut init_set = self.buffer_init.lock();
            let mut recreate = true;
            if let Some(buf) = cache.get(key) {
                if buf.len() == ((n_in / 16) * n_out) as u64 {
                    recreate = false;
                }
            }
            if recreate {
                let buf = Buffer::new_slice::<u32>(
                    self.memory_allocator.clone(),
                    BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
                    AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, ..Default::default() },
                    ((n_in / 16) * n_out) as u64,
                )?;
                let arc_buf = Arc::new(buf);
                // Write weights only on first allocation
                {
                    let weights_slice = unsafe { std::slice::from_raw_parts(packed_w, (n_in / 16) * n_out) };
                    let mut write_guard = arc_buf.write()?;
                    let copy_len = weights_slice.len().min(write_guard.len());
                    write_guard[..copy_len].copy_from_slice(&weights_slice[..copy_len]);
                }
                init_set.insert(key.to_string());
                cache.insert(key.to_string(), arc_buf.clone());
                arc_buf
            } else {
                cache.get(key).unwrap().clone()
            }
        };

        // Use cached output buffer if available
        let buffer_y = {
            let mut cache = self.buffer_y_cache.lock();
            let mut recreate = true;
            if let Some(buf) = cache.get(key) {
                if buf.len() == n_out as u64 {
                    recreate = false;
                }
            }
            if recreate {
                let buf = Buffer::new_slice::<f32>(
                    self.memory_allocator.clone(),
                    BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
                    AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_RANDOM_ACCESS, ..Default::default() },
                    n_out as u64,
                )?;
                let arc_buf = Arc::new(buf);
                cache.insert(key.to_string(), arc_buf.clone());
                arc_buf
            } else {
                cache.get(key).unwrap().clone()
            }
        };

        let layout = self.pipeline.layout().set_layouts().first().unwrap();
        let set = PersistentDescriptorSet::new(
            &*self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, (*buffer_x).clone()),
                WriteDescriptorSet::buffer(1, (*buffer_w).clone()),
                WriteDescriptorSet::buffer(2, (*buffer_y).clone()),
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
            .push_constants(self.pipeline.layout().clone(), 0, cs::PushConstants { n_in: n_in as u32, n_out: n_out as u32, scale })?
            .dispatch([n_out as u32, 1, 1])?; 

        let command_buffer = builder.build()?;
        sync::now(self.device.clone()).then_execute(self.queue.clone(), command_buffer)?.then_signal_fence_and_flush()?.wait(None)?;

        // Persistent host read directly from mapped memory buffer
        {
            let read_guard = buffer_y.read()?;
            let copy_len = y.len().min(read_guard.len());
            y[..copy_len].copy_from_slice(&read_guard[..copy_len]);
        }
        Ok(())
    }

    /// Executes a ternary GEMV on the iGPU (deprecated, use run_ternary_gemv_cached).
    pub fn run_ternary_gemv(
        &self,
        n_in: usize,
        n_out: usize,
        x: &[f32],
        packed_w: *const u32,
        scale: f32,
        y: &mut [f32],
    ) -> anyhow::Result<()> {
        let key = format!("ptr_{:x}", packed_w as usize);
        self.run_ternary_gemv_cached(&key, n_in, n_out, x, packed_w, scale, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulkan_init() {
        match VulkanContext::new() {
            Ok(ctx) => {
                println!("Vulkan initialized successfully on: {}", ctx.device.physical_device().properties().device_name);
            }
            Err(e) => {
                panic!("Failed to initialize Vulkan: {:?}", e);
            }
        }
    }
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/ternary_gemv.comp",
        vulkan_version: "1.1",
    }
}

