use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::{
    allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
};
use vulkano::descriptor_set::{
    allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::{
    compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::sync::{self, GpuFuture};

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "assets/shaders/elut_gemv_i16.comp",
        vulkan_version: "1.2"
    }
}

fn main() {
    println!("Initializing Vulkan SlimeShader Benchmark...");

    let library = vulkano::VulkanLibrary::new().expect("No local Vulkan library");
    let instance = Instance::new(library, InstanceCreateInfo::default()).unwrap();

    let physical_device = instance
        .enumerate_physical_devices()
        .unwrap()
        .next()
        .expect("No physical device found");

    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_, q)| q.queue_flags.intersects(QueueFlags::COMPUTE))
        .expect("No compute queue family") as u32;

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());
    let descriptor_set_allocator =
        StandardDescriptorSetAllocator::new(device.clone(), Default::default());

    // Compile shader
    let shader = cs::load(device.clone()).unwrap();
    let entry_point = shader.entry_point("main").unwrap();

    let stage = PipelineShaderStageCreateInfo::new(entry_point.clone());
    let pipeline = {
        let layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage]);
        let layout = PipelineLayout::new(
            device.clone(),
            layout_info
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .unwrap()
    };

    let n_in = 4096;
    let n_out = 4096;

    // Allocate InputX (SlimeRegister -> uint = 4096 elements)
    let input_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        (0..n_in).map(|_| 0x0000_0001u32), // 1 in i16, 0 in JEPA
    )
    .unwrap();

    // Allocate Weights (ELUT 4-bit -> 4096 * 4096 / 8 uints)
    let weights_size = (n_in * n_out) / 8;
    let weights_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        (0..weights_size).map(|_| 0x11111111u32), // All +1 weights
    )
    .unwrap();

    // Allocate OutputY (SlimeRegister -> uint = 4096 elements)
    let output_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        (0..n_out).map(|_| 0x1234_0000u32), // JEPA bits preset to 0x1234
    )
    .unwrap();

    let layout = pipeline.layout().set_layouts().first().unwrap();
    let set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, input_buffer.clone()),
            WriteDescriptorSet::buffer(1, weights_buffer.clone()),
            WriteDescriptorSet::buffer(2, output_buffer.clone()),
        ],
        [],
    )
    .unwrap();

    let push_constants = cs::PushConstants {
        n_in,
        n_out,
        do_residual: 0,
    };

    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .bind_pipeline_compute(pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            set,
        )
        .unwrap()
        .push_constants(pipeline.layout().clone(), 0, push_constants)
        .unwrap()
        .dispatch([n_out / 32, 1, 1])
        .unwrap();

    let command_buffer = builder.build().unwrap();

    let start = std::time::Instant::now();
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();
    let duration = start.elapsed();

    let output_content = output_buffer.read().unwrap();

    // Assert structural integrity
    // Each row does 4096 MACs with +1 and +1 -> 4096 sum.
    // Clamp at 32767. So the accum part should be 4096 (0x1000).
    // JEPA part was preset to 0x1234.
    // Result should be 0x1234_1000.

    let first = output_content[0];
    assert_eq!(
        first, 0x1234_1000,
        "Vulkan Shader failed mathematically. Got {:08X}",
        first
    );

    println!("Success! Mathematical and structural integrity verified.");
    println!(
        "Throughput: {:.2} GigaMAC/s",
        (n_in * n_out) as f64 / duration.as_secs_f64() / 1e9
    );
}
