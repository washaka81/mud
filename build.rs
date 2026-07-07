fn main() {
    // Compila los archivos de ensamblador hand-written para el i7-1260p
    cc::Build::new()
        .file("src/asm/q4_0_gemv.s")
        .file("src/asm/rmsnorm.s")
        .file("src/asm/rope.s")
        .file("src/asm/ternary_gemv.s")
        .file("src/asm/slime_rmsnorm.s")
        .file("src/asm/ternary_pext.s")
        .file("src/asm/ternary_lut.s")
        .file("src/asm/ternary_gemv_4rows.s")
        .file("src/asm/ternary_gemm_batch4.s")
        .file("src/asm/math.s")
        .file("src/asm/silu.s")
        .file("src/asm/mamba.s")
        .file("src/asm/sgemm.s")
        .file("src/asm/adam_step.s") // Adam AVX2 optimizer step kernel
        .file("src/asm/elut_gemv.s")
        .file("src/asm/lm_head.s") // LM head AVX2 kernel (batch dot product + argmax)
        .file("src/asm/qat_step.s") // AVX2 QAT step kernel
        .file("src/asm/ternary_backward.s") // AVX2 QAT backward kernel
        .flag("-mavx2")
        .flag("-mfma")
        .flag("-mbmi2")
        .compile("forge_asm");

    println!("cargo:rerun-if-changed=src/asm/q4_0_gemv.s");
    println!("cargo:rerun-if-changed=src/asm/rmsnorm.s");
    println!("cargo:rerun-if-changed=src/asm/rope.s");
    println!("cargo:rerun-if-changed=src/asm/ternary_gemv.s");
    println!("cargo:rerun-if-changed=src/asm/slime_rmsnorm.s");
    println!("cargo:rerun-if-changed=src/asm/ternary_pext.s");
    println!("cargo:rerun-if-changed=src/asm/ternary_lut.s");
    println!("cargo:rerun-if-changed=src/asm/ternary_gemv_4rows.s");
    println!("cargo:rerun-if-changed=src/asm/ternary_gemm_batch4.s");
    println!("cargo:rerun-if-changed=src/asm/math.s");
    println!("cargo:rerun-if-changed=src/asm/silu.s");
    println!("cargo:rerun-if-changed=src/asm/mamba.s");
    println!("cargo:rerun-if-changed=src/asm/sgemm.s");
    println!("cargo:rerun-if-changed=src/asm/adam_step.s");
    println!("cargo:rerun-if-changed=src/asm/elut_gemv.s");
    println!("cargo:rerun-if-changed=src/asm/lm_head.s");
    println!("cargo:rerun-if-changed=src/asm/qat_step.s");
    println!("cargo:rerun-if-changed=src/asm/ternary_backward.s");
}
