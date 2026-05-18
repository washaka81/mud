use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
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
}

impl VulkanContext {
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
            .filter(|p| {
                p.properties().device_type == PhysicalDeviceType::IntegratedGpu
                    && p.properties().device_name.contains("Intel")
            })
            .next()
            .ok_or_else(|| anyhow::anyhow!("No se encontró iGPU Intel Iris Xe"))?;

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

        Ok(Self { 
            device, queue, memory_allocator,
            command_buffer_allocator, descriptor_set_allocator,
        })
    }

    /// Runs a simple compute shader test.
    pub fn run_test_compute(&self) -> anyhow::Result<()> {
        let shader = cs::load(self.device.clone())?;
        let entry_point = shader.entry_point("main").unwrap();
        
        let pipeline = ComputePipeline::new(
            self.device.clone(),
            None,
            vulkano::pipeline::compute::ComputePipelineCreateInfo::stage_layout(
                vulkano::pipeline::PipelineShaderStageCreateInfo::new(entry_point.clone()),
                PipelineLayout::new(
                    self.device.clone(),
                    PipelineDescriptorSetLayoutCreateInfo::from_stages([&vulkano::pipeline::PipelineShaderStageCreateInfo::new(entry_point.clone())])
                        .into_pipeline_layout_create_info(self.device.clone())?,
                )?,
            ),
        )?;

        let data_in = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
            AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, ..Default::default() },
            (0..1024).map(|i| i as f32),
        )?;

        let data_out = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
            AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_RANDOM_ACCESS, ..Default::default() },
            (0..1024).map(|_| 0.0f32),
        )?;

        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            &*self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, data_in.clone()),
                WriteDescriptorSet::buffer(1, data_out.clone()),
                WriteDescriptorSet::buffer(2, data_out.clone()),
            ],
            [],
        )?;

        let mut builder = AutoCommandBufferBuilder::primary(
            &*self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        builder
            .bind_pipeline_compute(pipeline.clone())?
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline.layout().clone(), 0, set)?
            .push_constants(pipeline.layout().clone(), 0, cs::PushConstants { n_in: 1024, n_out: 4, scale: 2.0 })?
            .dispatch([4, 1, 1])?; 

        let command_buffer = builder.build()?;
        sync::now(self.device.clone()).then_execute(self.queue.clone(), command_buffer)?.then_signal_fence_and_flush()?.wait(None)?;
        Ok(())
    }

    /// Executes a ternary GEMV on the iGPU.
    pub fn run_ternary_gemv(
        &self,
        n_in: usize,
        n_out: usize,
        x: &[f32],
        packed_w: *const u32,
        scale: f32,
        y: &mut [f32],
    ) -> anyhow::Result<()> {
        let shader = cs::load(self.device.clone())?;
        let entry_point = shader.entry_point("main").unwrap();
        
        let pipeline = ComputePipeline::new(
            self.device.clone(),
            None,
            vulkano::pipeline::compute::ComputePipelineCreateInfo::stage_layout(
                vulkano::pipeline::PipelineShaderStageCreateInfo::new(entry_point.clone()),
                PipelineLayout::new(
                    self.device.clone(),
                    PipelineDescriptorSetLayoutCreateInfo::from_stages([&vulkano::pipeline::PipelineShaderStageCreateInfo::new(entry_point.clone())])
                        .into_pipeline_layout_create_info(self.device.clone())?,
                )?,
            ),
        )?;

        let buffer_x = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
            AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, ..Default::default() },
            x.iter().cloned(),
        )?;

        let weights_slice = unsafe { std::slice::from_raw_parts(packed_w, (n_in / 16) * n_out) };
        let buffer_w = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
            AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, ..Default::default() },
            weights_slice.iter().cloned(),
        )?;

        let buffer_y = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
            AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_RANDOM_ACCESS, ..Default::default() },
            (0..y.len()).map(|_| 0.0f32),
        )?;

        let layout = pipeline.layout().set_layouts().get(0).unwrap();
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
            .bind_pipeline_compute(pipeline.clone())?
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline.layout().clone(), 0, set)?
            .push_constants(pipeline.layout().clone(), 0, cs::PushConstants { n_in: n_in as u32, n_out: n_out as u32, scale })?
            .dispatch([n_out as u32, 1, 1])?; 

        let command_buffer = builder.build()?;
        sync::now(self.device.clone()).then_execute(self.queue.clone(), command_buffer)?.then_signal_fence_and_flush()?.wait(None)?;

        let content = buffer_y.read()?;
        for i in 0..y.len() { if i < content.len() { y[i] = content[i]; } }
        Ok(())
    }
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/ternary_gemv.comp",
        vulkan_version: "1.1",
    }
}

