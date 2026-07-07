use forge_llm::asm::elut_gemv_avx2;
use std::time::Instant;

fn main() {
    let hidden_size = 2048; // Must be multiple of 32

    // Allocate 100MB of activations and weights to measure throughput
    let num_rows = 100_000;

    let activations = vec![1i8; hidden_size];
    // Weights in ELUT 4-bit nibbles:
    // 0x11 = 1 and 1
    // 0xFF = -1 and -1
    // 0x1F = 1 and -1
    let weights = vec![0x11u8; hidden_size / 2 * num_rows];
    let mut accumulators = vec![0i16; num_rows];

    println!("ELUT-AVX2 Kernel Benchmark");
    println!("Matrix: {}x{}", num_rows, hidden_size);

    let start = Instant::now();

    for i in 0..num_rows {
        unsafe {
            elut_gemv_avx2(
                activations.as_ptr(),
                weights.as_ptr().add(i * (hidden_size / 2)),
                accumulators.as_mut_ptr().add(i),
                hidden_size,
            );
        }
    }

    let duration = start.elapsed();
    let ops = (hidden_size as f64) * (num_rows as f64);
    let throughput = ops / duration.as_secs_f64();
    let memory_bytes = ops * 1.5; // 1 byte act, 0.5 byte weight
    let bandwidth = memory_bytes / duration.as_secs_f64() / 1e9;

    println!("Time: {:.2?}", duration);
    println!("Throughput: {:.2} elements/sec", throughput);
    println!("Bandwidth: {:.2} GB/s", bandwidth);

    // Simple correctness check
    assert_eq!(accumulators[0], 2048);
}
