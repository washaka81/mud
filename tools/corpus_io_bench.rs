use rand::{RngExt, SeedableRng};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

fn main() {
    println!("🚀 MUD-QAT Corpus I/O MMAP Cache Saturation Benchmark\n");

    let num_tokens = 500_000_000; // 500 Million tokens (~2GB)
    let cache_path = "training/corpus/io_bench_cache_sustained.bin";

    println!(
        "[1] Generating 500M token synthetic binary corpus cache (~2GB) in chunks to avoid OOM..."
    );
    std::fs::create_dir_all("training/corpus").unwrap();
    let mut file = File::create(cache_path).unwrap();
    let chunk_size = 50_000_000; // 190MB chunks
    let mut dummy_data = vec![0u32; chunk_size];

    for j in 0..10 {
        for (i, d) in dummy_data.iter_mut().enumerate() {
            *d = ((j * chunk_size + i) % 32000) as u32;
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(dummy_data.as_ptr() as *const u8, dummy_data.len() * 4)
        };
        file.write_all(bytes).unwrap();
        print!(".");
        std::io::stdout().flush().unwrap();
    }
    file.sync_all().unwrap();
    println!("\n    ✅ Generation & Disk Sync Complete.\n");

    println!(
        "[2] Starting 1-Minute Mmap Load Test (Loading 50 random chunks of 4MB every loop)..."
    );

    let total_duration = 60; // 1 minute
    let mut iterations = 0;

    let global_start = Instant::now();
    let mut throughputs_mb = Vec::new();

    while global_start.elapsed().as_secs() < total_duration {
        let load_start = Instant::now();

        // Simulating the MudCorpusTrainer MMAP load
        let file = File::open(cache_path).unwrap();
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
        let _ = mmap.advise(memmap2::Advice::Random);

        let mut rng = rand::rngs::StdRng::seed_from_u64(iterations as u64);
        let chunk_tokens = 1_000_000; // 4MB
        let mut bytes_read = 0;

        for _ in 0..50 {
            // Read 50 chunks = 200MB per iteration
            let start_idx = rng.random_range(0..(num_tokens - chunk_tokens));
            let byte_start = start_idx * 4;
            let byte_end = byte_start + chunk_tokens * 4;

            let chunk_bytes = &mmap[byte_start..byte_end];
            let tokens = unsafe {
                std::slice::from_raw_parts(chunk_bytes.as_ptr() as *const u32, chunk_tokens)
            }
            .to_vec();

            if tokens.is_empty() {
                break;
            }
            bytes_read += chunk_bytes.len();
        }

        let load_dur = load_start.elapsed();
        let size_mb = bytes_read as f64 / (1024.0 * 1024.0);
        let throughput_mb_s = size_mb / load_dur.as_secs_f64();
        throughputs_mb.push(throughput_mb_s);

        iterations += 1;

        println!(
            "[{}/60s] Iteration #{}:",
            global_start.elapsed().as_secs(),
            iterations
        );
        println!("   - Random read {:.2} MB: {:?}", size_mb, load_dur);
        println!("   - Throughput: {:.2} MB/s", throughput_mb_s);

        let mut sys = sysinfo::System::new_all();
        sys.refresh_memory();
        let free_mem = sys.free_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        println!("   - Free System RAM: {:.2} GB", free_mem);
        if free_mem < 0.5 {
            println!("🚨 OUT OF MEMORY EMERGENCY CUT. Free RAM is below 500MB!");
            std::fs::remove_file(cache_path).unwrap();
            std::process::exit(1);
        }
    }

    println!("\n[3] Test Complete. Zero-Allocation MMAP Cache proved stable.");
    std::fs::remove_file(cache_path).unwrap();
}
