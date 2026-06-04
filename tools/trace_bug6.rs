use forge_llm::mud::MudFile;

fn main() {
    println!("🔍 INICIANDO TRAZA BUG-6...");
    let model = MudFile::load("models/core_skills.mud").expect("Failed to load mud");
    let core = model.skills.get("core").unwrap();

    let w1 = core
        .tensors
        .get("blk.0.expert.0.w1.weight")
        .expect("w1 missing");
    let s1 = core.tensors.get("blk.0.expert.0.w1.prq_scale");

    println!("=> Tensor w1 shape: {:?}", w1.shape);
    let elements: usize = w1.shape.iter().product();
    println!("=> w1 elements: {}", elements);

    let u32_count = elements.div_ceil(16);
    let packed_data = unsafe { std::slice::from_raw_parts(w1.data_ptr as *const u32, u32_count) };

    let mut counts = [0usize; 3];
    let mut first_few = Vec::new();

    for (idx, &val) in packed_data.iter().enumerate() {
        for i in 0..16 {
            if idx == 0 && i < 16 {
                let bits = (val >> (i * 2)) & 3;
                first_few.push(bits);
            }
            let bits = (val >> (i * 2)) & 3;
            if bits == 1 {
                counts[1] += 1;
            } else if bits == 2 {
                counts[2] += 1;
            } else {
                counts[0] += 1;
            }
        }
    }

    let total = counts[0] + counts[1] + counts[2];
    println!(
        "=> Distribution: 0s: {} ({:.2}%), 1s: {} ({:.2}%), 2s (-1): {} ({:.2}%)",
        counts[0],
        (counts[0] as f32 / total as f32) * 100.0,
        counts[1],
        (counts[1] as f32 / total as f32) * 100.0,
        counts[2],
        (counts[2] as f32 / total as f32) * 100.0
    );
    println!("=> First 16 unpacked bits: {:?}", first_few);

    if let Some(scale) = s1 {
        let rows = if w1.shape.len() == 2 { w1.shape[0] } else { 1 };
        let scales = unsafe { std::slice::from_raw_parts(scale.data_ptr as *const f32, rows) };
        println!(
            "=> Found .scale tensor! First 5 scales: {:?}",
            &scales[0..5.min(scales.len())]
        );
    } else {
        println!("=> NO .scale tensor found!");
    }
}
