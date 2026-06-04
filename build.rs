fn main() {
    // Compila los archivos de ensamblador hand-written para el i7-1260p
    cc::Build::new()
        .file("src/asm/q4_0_gemv.s")
        .file("src/asm/rmsnorm.s")
        .file("src/asm/rope.s")
        .file("src/asm/ternary_gemv.s")
        .file("src/asm/ternary_pext.s")
        .file("src/asm/ternary_lut.s")
        .file("src/asm/ternary_gemv_4rows.s")
        .file("src/asm/ternary_gemm_batch4.s")
        .file("src/asm/math.s")
        .file("src/asm/silu.s")
        .file("src/asm/mamba.s")
        .flag("-march=native")
        .compile("forge_asm");

    println!("cargo:rerun-if-changed=src/asm/q4_0_gemv.s");
    println!("cargo:rerun-if-changed=src/asm/rmsnorm.s");
    println!("cargo:rerun-if-changed=src/asm/rope.s");
    println!("cargo:rerun-if-changed=src/asm/ternary_gemv.s");
    println!("cargo:rerun-if-changed=src/asm/ternary_pext.s");
    println!("cargo:rerun-if-changed=src/asm/ternary_lut.s");
    println!("cargo:rerun-if-changed=src/asm/ternary_gemv_4rows.s");
    println!("cargo:rerun-if-changed=src/asm/ternary_gemm_batch4.s");
    println!("cargo:rerun-if-changed=src/asm/math.s");
    println!("cargo:rerun-if-changed=src/asm/silu.s");
    println!("cargo:rerun-if-changed=src/asm/mamba.s");
}
