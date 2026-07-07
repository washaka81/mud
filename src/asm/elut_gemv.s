.section .text
.global elut_gemv_avx2

# elut_gemv_avx2(activations: *const i8, weights_elut: *const u8, accumulators: *mut i16, n: usize)
#
# rdi: activations (i8*)
# rsi: weights_elut (u8*)
# rdx: accumulators (i16*)
# rcx: n (usize)

elut_gemv_avx2:
    push %rbp
    mov %rsp, %rbp

    # ymm15: accumulator for the row (i32)
    vxorps %ymm15, %ymm15, %ymm15
    vxorps %ymm14, %ymm14, %ymm14

    # ymm13: partial i16 accumulator
    vxorps %ymm13, %ymm13, %ymm13

    # ymm12: ELUT decoding LUT
    # 0 -> 0
    # 1 -> 1
    # 15 (0xF) -> -1 (0xFF)
    movabs $0x0000000000000100, %r8
    vmovq %r8, %xmm12
    movabs $0xFF00000000000000, %r8
    vpinsrq $1, %r8, %xmm12, %xmm12
    vinserti128 $1, %xmm12, %ymm12, %ymm12

    # ymm11: Mask 0x0F
    mov $0x0F0F0F0F, %eax
    vmovd %eax, %xmm11
    vpbroadcastd %xmm11, %ymm11

    # ymm10: Ones for vpmaddubsw
    mov $0x01010101, %eax
    vmovd %eax, %xmm10
    vpbroadcastd %xmm10, %ymm10

    mov %rcx, %r8
    mov $0, %r9

.loop:
    cmp $32, %r8
    jl .done_loop

    # Load 16 bytes ELUT
    vmovdqu (%rsi), %xmm1
    vpmovzxbw %xmm1, %ymm1
    
    # Unpack nibbles
    vpand %ymm11, %ymm1, %ymm2
    vpsrlw $4, %ymm1, %ymm3
    vpand %ymm11, %ymm3, %ymm3
    
    # LUT map
    vpshufb %ymm2, %ymm12, %ymm2
    vpshufb %ymm3, %ymm12, %ymm3

    # Interleave to 32 bytes of weights
    vpsllw $8, %ymm3, %ymm3
    vpor %ymm3, %ymm2, %ymm1

    # Load activations
    vmovdqu (%rdi), %ymm0

    # Multiply A * W using vpsignb
    # vpsignb negates ymm0 where ymm1 is negative, zeros where ymm1 is 0
    vpsignb %ymm1, %ymm0, %ymm0

    # Horizontal add pairs to i16 using vpmaddubsw with 1s
    # Note: vpmaddubsw requires first op to be unsigned.
    # The 1s are unsigned (ymm10). The ymm0 are signed.
    vpmaddubsw %ymm0, %ymm10, %ymm0

    # Accumulate into partial i16
    vpaddw %ymm0, %ymm13, %ymm13

    # Reseat check
    add $32, %r9
    cmp $256, %r9
    jl .skip_reseat

    # Reseat: vpmovsxwd to i32, accumulate to ymm14/15, zero ymm13
    vpmovsxwd %xmm13, %ymm4
    vextracti128 $1, %ymm13, %xmm13
    vpmovsxwd %xmm13, %ymm5
    vpaddd %ymm4, %ymm14, %ymm14
    vpaddd %ymm5, %ymm15, %ymm15
    vxorps %ymm13, %ymm13, %ymm13
    mov $0, %r9

.skip_reseat:
    add $32, %rdi
    add $16, %rsi
    sub $32, %r8
    jmp .loop

.done_loop:
    # Final reseat of whatever is left in ymm13
    vpmovsxwd %xmm13, %ymm4
    vextracti128 $1, %ymm13, %xmm13
    vpmovsxwd %xmm13, %ymm5
    vpaddd %ymm4, %ymm14, %ymm14
    vpaddd %ymm5, %ymm15, %ymm15

    # Horizontal reduction of i32
    vpaddd %ymm15, %ymm14, %ymm14
    vextracti128 $1, %ymm14, %xmm15
    vpaddd %xmm15, %xmm14, %xmm14
    vphaddd %xmm14, %xmm14, %xmm14
    vphaddd %xmm14, %xmm14, %xmm14

    # We have the 32-bit sum in the lowest dword of xmm14.
    # We must write to *accumulators as i16.
    # The mandate: "Accumulate into i16". 
    # But wait, earlier I said it's a dot product of ONE row.
    # If the user passes *mut i16, we write a single i16 scalar?
    # No, typically GEMV processes multiple rows, or we loop in Rust.
    # Let's assume it writes a single i16 scalar.
    vmovd %xmm14, %eax
    movw %ax, (%rdx)

    pop %rbp
    ret
