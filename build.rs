fn main() {
    // Handwritten AVX2 ASM for i7-1260P hot paths (L-04: purged obsolete orphans)
    cc::Build::new()
        .file("src/asm/q4_0_gemv.s")
        .file("src/asm/rmsnorm.s")
        .file("src/asm/rope.s")
        .file("src/asm/ternary_gemv.s")
        .file("src/asm/ternary_gemv_4rows.s")
        .file("src/asm/ternary_gemv_8rows.s")
        .file("src/asm/ternary_gemm_batch4.s")
        .file("src/asm/math.s")
        .file("src/asm/silu.s")
        .file("src/asm/sgemm.s")
        .file("src/asm/adam_step.s") // reserved for full Adam moments path
        .file("src/asm/lm_head.s")
        .flag("-mavx2")
        .flag("-mfma")
        .flag("-mbmi2")
        .compile("forge_asm");

    for f in [
        "src/asm/q4_0_gemv.s",
        "src/asm/rmsnorm.s",
        "src/asm/rope.s",
        "src/asm/ternary_gemv.s",
        "src/asm/ternary_gemv_4rows.s",
        "src/asm/ternary_gemv_8rows.s",
        "src/asm/ternary_gemm_batch4.s",
        "src/asm/math.s",
        "src/asm/silu.s",
        "src/asm/sgemm.s",
        "src/asm/adam_step.s",
        "src/asm/lm_head.s",
    ] {
        println!("cargo:rerun-if-changed={f}");
    }

    // Phase 15: Pre-compile GLSL Compute Shaders → SPIR-V for ash backend
    compile_spirv_shaders();
}

/// Compile all GLSL compute shaders to SPIR-V using glslc.
/// Skips gracefully if glslc is not installed.
fn compile_spirv_shaders() {
    use std::path::Path;
    use std::process::Command;

    let shaders = [
        (
            "assets/shaders/ternary_gemv_unified.comp",
            "assets/shaders/spirv/ternary_gemv_unified.spv",
        ),
        (
            "assets/shaders/silu_gate.comp",
            "assets/shaders/spirv/silu_gate.spv",
        ),
        (
            "assets/shaders/shadow_optimizer.comp",
            "assets/shaders/spirv/shadow_optimizer.spv",
        ),
        (
            "assets/shaders/ternary_backward.comp",
            "assets/shaders/spirv/ternary_backward.spv",
        ),
        (
            "assets/shaders/newton_schulz_step1.comp",
            "assets/shaders/spirv/newton_schulz_step1.spv",
        ),
        (
            "assets/shaders/newton_schulz_step2.comp",
            "assets/shaders/spirv/newton_schulz_step2.spv",
        ),
        (
            "assets/shaders/tensor_thermodynamics.comp",
            "assets/shaders/spirv/tensor_thermodynamics.spv",
        ),
        (
            "assets/shaders/heartbeat.comp",
            "assets/shaders/spirv/heartbeat.spv",
        ),
        (
            "assets/shaders/rms_norm.comp",
            "assets/shaders/spirv/rms_norm.spv",
        ),
        ("assets/shaders/mha.comp", "assets/shaders/spirv/mha.spv"),
    ];

    if Command::new("glslc").arg("--version").output().is_err() {
        println!("cargo:warning=glslc not found — ash backend SPIR-V compilation skipped. Install Vulkan SDK to enable Phase 15.");
        return;
    }

    std::fs::create_dir_all("assets/shaders/spirv")
        .expect("Failed to create SPIR-V output directory");

    for (src, dst) in &shaders {
        let needs_compile = if Path::new(dst).exists() {
            let src_time = std::fs::metadata(src).and_then(|m| m.modified()).ok();
            let dst_time = std::fs::metadata(dst).and_then(|m| m.modified()).ok();
            match (src_time, dst_time) {
                (Some(s), Some(d)) => s > d,
                _ => true,
            }
        } else {
            true
        };

        if needs_compile {
            let status = Command::new("glslc")
                .args(["--target-env=vulkan1.1", "-O", src, "-o", dst])
                .status()
                .expect("glslc failed to launch");

            if !status.success() {
                panic!("SPIR-V compilation failed for: {src}");
            }
            println!("cargo:warning=Compiled SPIR-V: {src} → {dst}");
        }

        println!("cargo:rerun-if-changed={src}");
    }
}
