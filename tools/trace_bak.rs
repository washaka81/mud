use forge_llm::mud::MudFile;

fn main() {
    println!("🔍 INICIANDO TRAZA CORE SKILLS BAK...");
    let model = MudFile::load("models/core_skills.mud.bak").expect("Failed to load mud");
    let core = model.skills.get("core").unwrap();

    let w1 = core
        .tensors
        .get("blk.0.expert.0.w1.weight")
        .expect("w1 missing");
    let s1 = core.tensors.get("blk.0.expert.0.w1.scale");

    let elements: usize = w1.shape.iter().product();
    let u32_count = elements.div_ceil(16);
    let packed_data = unsafe { std::slice::from_raw_parts(w1.data_ptr as *const u32, u32_count) };

    let mut counts = [0usize; 3];
    for &val in packed_data {
        for i in 0..16 {
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
        "=> BAK Distribution: 0s: {} ({:.2}%), 1s: {} ({:.2}%), 2s (-1): {} ({:.2}%)",
        counts[0],
        (counts[0] as f32 / total as f32) * 100.0,
        counts[1],
        (counts[1] as f32 / total as f32) * 100.0,
        counts[2],
        (counts[2] as f32 / total as f32) * 100.0
    );

    if let Some(_scale) = s1 {
        println!("=> BAK has .scale tensor!");
    } else {
        println!("=> BAK has NO .scale tensor!");
    }
}
