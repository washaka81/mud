use std::time::Instant;

fn main() {
    println!("🚀 [EZOP] Engine Zero-Overhead Protocol Validator & Benchmark\n");

    let elements = 10_000_000;
    println!(
        "Target Array Size: {} elements ({} MB)",
        elements,
        (elements * 4) / 1024 / 1024
    );

    let mut data_slice = vec![1.0f32; elements];
    let mut data_ezop = vec![1.0f32; elements];
    let grad = vec![0.01f32; elements];

    println!("\n[1] Standard Rust Slices (Bounds Checking + Iterator Overhead)");
    let start_slice = Instant::now();
    for i in 0..elements {
        data_slice[i] -= 0.001 * grad[i];
    }
    let dur_slice = start_slice.elapsed();
    let ops_slice = elements as f64 / dur_slice.as_secs_f64();
    println!("    Latency: {:?}", dur_slice);
    println!("    Throughput: {:.2} elements/sec", ops_slice);

    println!("\n[2] EZOP Protocol (Pure Raw Pointers *mut T)");
    let start_ezop = Instant::now();
    let ptr_ezop = data_ezop.as_mut_ptr();
    let ptr_grad = grad.as_ptr();
    unsafe {
        for i in 0..elements {
            *ptr_ezop.add(i) = *ptr_ezop.add(i) - 0.001 * (*ptr_grad.add(i));
        }
    }
    let dur_ezop = start_ezop.elapsed();
    let ops_ezop = elements as f64 / dur_ezop.as_secs_f64();
    println!("    Latency: {:?}", dur_ezop);
    println!("    Throughput: {:.2} elements/sec", ops_ezop);

    println!("\n[3] Mathematical Equivalence Verification");
    let mut divergence = 0.0;
    for i in 0..elements {
        let diff = (data_slice[i] - data_ezop[i]).abs();
        if diff > divergence {
            divergence = diff;
        }
    }
    println!("    Max Divergence: {:.10}", divergence);
    if divergence == 0.0 {
        println!("    ✅ PASS: EZOP math is perfectly identical to Safe Rust.");
    } else {
        println!("    ❌ FAIL: EZOP deviated from safe execution!");
        std::process::exit(1);
    }

    let speedup = ops_ezop / ops_slice;
    println!("\n🏆 EZOP VALIDATION COMPLETE.");
    println!("   Speedup Factor: {:.2}x", speedup);
}
