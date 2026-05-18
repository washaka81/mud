use cc;

fn main() {
    // Compila los archivos de ensamblador hand-written para el i7-1260p
    cc::Build::new()
        .file("src/asm/q4_0_gemv.s")
        .file("src/asm/q4_0_gemv_fused.s")
        .file("src/asm/rmsnorm.s")
        .file("src/asm/rope.s")
        .file("src/asm/ternary_gemv.s")
        .flag("-march=native")
        .compile("forge_asm");

    println!("cargo:rerun-if-changed=src/asm/q4_0_gemv.s");
    println!("cargo:rerun-if-changed=src/asm/ternary_gemv.s");
}
