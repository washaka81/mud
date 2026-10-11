.section .text

# ── registers_to_f32_batch ──────────────────────────────────────
# Convert N PackedRegister (u32, f16×2) to separate real/imag f32 arrays.
#
# C signature:
#   void registers_to_f32_batch(
#       const uint32_t *regs,    // %rdi
#       float *out_real,         // %rsi
#       float *out_imag,         // %rdx
#       size_t n                 // %rcx — must be multiple of 8
#   );
#
# AVX2 (F16C required): vcvtph2ps on i7-1260P has 1-cycle throughput.
# Speedup vs scalar: ~4× (6 instr per 8 regs vs 32+ instr).
.global registers_to_f32_batch
registers_to_f32_batch:
    push %rbp
    mov %rsp, %rbp
    xor %rax, %rax          # i = 0
.Lr2f_loop:
    cmp %rcx, %rax
    jae .Lr2f_done
    vmovdqu (%rdi, %rax, 4), %ymm0      # 8 packed regs (256 bits)
    vextracti128 $0, %ymm0, %xmm1       # low 128b  = real f16 ×4 + imag f16 ×4 = 2+6 slots
    # Actually each 32-bit word = [imag(f16) | real(f16)]
    # xmm1 = { re0, im0, re1, im1, re2, im2, re3, im3 } as 8×16-bit
    vextracti128 $1, %ymm0, %xmm2       # xmm2 = { re4, im4, re5, im5, re6, im6, re7, im7 }
    
    # Extract real: need byte pairs 0,2,4,6 from each xmm
    # Use vpshufb with mask to pick the low 16 bits of each dword (the real f16)
    # Real values are in dword positions 0,1,2,3 (low half), 4,5,6,7 (high half)
    # ymm0 real bytes: (0,1),(4,5),(8,9),(12,13) — from the full 256-bit reg
    # But we separated into xmm1 and xmm2, so for each:
    # real bytes: (0,1),(4,5),(8,9),(12,13)
    # Same mask works per-lane
    
    # First handle real: extract from xmm1 and xmm2
    # Since xmm1 = {re0, im0, re1, im1, re2, im2, re3, im3}
    # Use vpsrlq $32 to shift each qword right... no, that's complex
    
    # Simpler: use vpshufb on full ymm to extract real bytes
    vmovdqa .Lreal_extract_mask(%rip), %ymm3
    vpshufb %ymm3, %ymm0, %ymm3          # ymm3 lane 0: {re0[7:0],re0[15:8],0,0,re1[7:0],re1[15:8],0,0,...}
                                          # lane 1: {re4,re5,re6,re7}
    
    # Now compress each lane's 4 real f16 into contiguous bytes
    # lane 0: bytes 0,1 = re0, bytes 4,5 = re1, bytes 8,9 = re2, bytes 12,13 = re3
    # We want re0, re1, re2, re3 contiguous in low 64 bits of lane 0
    vmovdqa .Lreal_compress_mask(%rip), %ymm4
    vpshufb %ymm4, %ymm3, %ymm4          # lane 0: {re0, re1, re2, re3, zeros...}
                                          # lane 1: {re4, re5, re6, re7, zeros...}
    
    vextracti128 $0, %ymm4, %xmm5        # xmm5 = {re0, re1, re2, re3, zeros}
    vextracti128 $1, %ymm4, %xmm6        # xmm6 = {re4, re5, re6, re7, zeros}
    vpunpcklqdq %xmm6, %xmm5, %xmm5      # xmm5 = {re0, re1, re2, re3, re4, re5, re6, re7} as 8 f16
    vcvtph2ps %xmm5, %ymm5               # ymm5 = f32(re0..re7)
    vmovdqu %ymm5, (%rsi, %rax, 4)       # store real f32
    
    # Same for imag: extract from ymm0 using imag mask
    # imag sits in bytes 2,3,6,7,10,11,14,15 of each 16-byte sequence
    vmovdqa .Limag_extract_mask(%rip), %ymm3
    vpshufb %ymm3, %ymm0, %ymm3          # extract imag to low byte of each lane dword
    vmovdqa .Limag_compress_mask(%rip), %ymm4
    vpshufb %ymm4, %ymm3, %ymm4          # compress
    vextracti128 $0, %ymm4, %xmm5
    vextracti128 $1, %ymm4, %xmm6
    vpunpcklqdq %xmm6, %xmm5, %xmm5
    vcvtph2ps %xmm5, %ymm5               # ymm5 = f32(im0..im7)
    vmovdqu %ymm5, (%rdx, %rax, 4)       # store imag f32
    
    add $8, %rax
    jmp .Lr2f_loop
.Lr2f_done:
    vzeroupper
    pop %rbp
    ret


# ── f32_to_registers_batch ──────────────────────────────────────
# Pack 8 real + 8 imag f32 values into 8 PackedRegister (u32, f16×2).
#
# C signature:
#   void f32_to_registers_batch(
#       const float *real,       // %rdi
#       const float *imag,       // %rsi
#       uint32_t *out_regs,      // %rdx
#       size_t n                 // %rcx — must be multiple of 8
#   );
.global f32_to_registers_batch
f32_to_registers_batch:
    push %rbp
    mov %rsp, %rbp
    xor %rax, %rax          # i = 0
.Lf2r_loop:
    cmp %rcx, %rax
    jae .Lf2r_done
    vmovups (%rdi, %rax, 4), %ymm0      # 8 real f32
    vmovups (%rsi, %rax, 4), %ymm1      # 8 imag f32
    vcvtps2ph $0, %ymm0, %xmm2          # xmm2 = 8 real f16 packed
    vcvtps2ph $0, %ymm1, %xmm3          # xmm3 = 8 imag f16 packed
    # Interleave: xmm2 = {re0, re1, re2, re3, re4, re5, re6, re7}
    #             xmm3 = {im0, im1, im2, im3, im4, im5, im6, im7}
    # Want: low 64b = {re0, im0, re1, im1} = 4 × 32-bit packed regs = 128b
    #       high 64b = {re4, im4, re5, im5} = next 4 packed regs = 128b
    # Using vpunpcklwd/hwd (interleave words within each 128-bit lane):
    vpunpcklwd %xmm3, %xmm2, %xmm4      # xmm4 = {re0, im0, re1, im1, re2, im2, re3, im3}
    vpunpckhwd %xmm3, %xmm2, %xmm5      # xmm5 = {re4, im4, re5, im5, re6, im6, re7, im7}
    vinserti128 $1, %xmm5, %ymm4, %ymm0 # ymm0 = 8 packed regs
    vmovdqa %ymm0, (%rdx, %rax, 4)       # store
    add $8, %rax
    jmp .Lf2r_loop
.Lf2r_done:
    vzeroupper
    pop %rbp
    ret


# ── gate_batch ──────────────────────────────────────────────────
# Compute sigmoid gate for 8 packed registers (imag component).
# gate = 1.0 / (1.0 + exp(-imag_f16->f32))
#
# C signature:
#   void gate_batch(
#       const uint32_t *regs,    // %rdi
#       float *out_gates,        // %rsi
#       size_t n                 // %rdx — must be multiple of 8
#   );
.global gate_batch
gate_batch:
    push %rbp
    mov %rsp, %rbp
    
    # Load constants
    vmovaps .Llog2e(%rip), %ymm8
    vmovaps .Lc4(%rip), %ymm9
    vmovaps .Lc3(%rip), %ymm10
    vmovaps .Lc2(%rip), %ymm11
    vmovaps .Lc1(%rip), %ymm12
    vmovaps .Lc0(%rip), %ymm13
    vmovdqa .Li127(%rip), %ymm14
    vmovaps .Lneg_zero(%rip), %ymm15
    vmovaps .Ltwo(%rip), %ymm7
    
    xor %rax, %rax          # i = 0
.Lgate_loop:
    cmp %rdx, %rax
    jae .Lgate_done
    
    vmovdqa (%rdi, %rax, 4), %ymm0      # 8 packed regs
    
    # Extract imag f16 using native [im:re] layout (im in bits 16-31 = bytes 2,3).
    # A-1 FIX (2026-08-03): the previous `vpsrld $16` moved im to bits 0-15,
    # leaving bytes 2,3 = 0 — the extract mask then selected zeros, producing
    # gate = sigmoid(0) = 0.5 CONSTANT for all inputs. Simply copy and let
    # vpshufb pick bytes 2,3,6,7,10,11,14,15 (im) directly from the native layout.
    vmovdqa %ymm0, %ymm1            # copy 8 packed regs (work on reg 1)
    vmovdqa .Limag_extract_mask(%rip), %ymm2
    vpshufb %ymm2, %ymm1, %ymm1     # compress: picks byte pairs 2,3 / 6,7 / ... = im f16 values
    vextracti128 $0, %ymm1, %xmm2
    vextracti128 $1, %ymm1, %xmm3
    vpunpcklqdq %xmm3, %xmm2, %xmm2     # xmm2 = 8 imag f16 packed
    vcvtph2ps %xmm2, %ymm0              # ymm0 = f32(imag[0..7])
    
    # Clip for stable exp
    vmaxps .Lclamp_lo(%rip), %ymm0, %ymm0
    vminps .Lclamp_hi(%rip), %ymm0, %ymm0
    
    # Compute sigmoid: 1/(1+exp(-x))
    vxorps %ymm15, %ymm0, %ymm1         # ymm1 = -x
    vmulps %ymm8, %ymm1, %ymm2          # y = -x * log2e
    vroundps $0, %ymm2, %ymm3           # n = round(y)
    vsubps %ymm3, %ymm2, %ymm4          # f = y - n
    
    vmovaps %ymm9, %ymm5
    vfmadd213ps %ymm10, %ymm4, %ymm5    # p = c4*f + c3
    vfmadd213ps %ymm11, %ymm4, %ymm5    # p = p*f + c2
    vfmadd213ps %ymm12, %ymm4, %ymm5    # p = p*f + c1
    vfmadd213ps %ymm13, %ymm4, %ymm5    # p = p*f + 1 = 2^f
    
    vcvtps2dq %ymm3, %ymm6
    vpaddd %ymm14, %ymm6, %ymm6
    vpslld $23, %ymm6, %ymm6
    vmulps %ymm6, %ymm5, %ymm5          # ymm5 = exp(-x)
    
    vaddps %ymm13, %ymm5, %ymm5         # d = 1 + exp(-x)
    vrcpps %ymm5, %ymm6
    vmulps %ymm6, %ymm5, %ymm1
    vsubps %ymm1, %ymm7, %ymm1
    vmulps %ymm1, %ymm6, %ymm6          # gate = 1/d (NR refined)
    
    vmovups %ymm6, (%rsi, %rax, 4)      # store 8 gates
    add $8, %rax
    jmp .Lgate_loop
.Lgate_done:
    vzeroupper
    pop %rbp
    ret


.section .rodata
.align 32

# Extract real bytes from packed reg: pick byte pairs 0,1 for re0; 4,5 for re1; 8,9 for re2; 12,13 for re3
.Lreal_extract_mask:
.byte 0, 1, 0x80, 0x80, 4, 5, 0x80, 0x80, 8, 9, 0x80, 0x80, 12, 13, 0x80, 0x80
.byte 0, 1, 0x80, 0x80, 4, 5, 0x80, 0x80, 8, 9, 0x80, 0x80, 12, 13, 0x80, 0x80

# Compress real: pick bytes 0,1,4,5,8,9,12,13 from the extract mask output
.Lreal_compress_mask:
.byte 0, 1, 4, 5, 8, 9, 12, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
.byte 0, 1, 4, 5, 8, 9, 12, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80

# Extract imag bytes: pick byte pairs 2,3 for im0; 6,7 for im1; 10,11 for im2; 14,15 for im3
.Limag_extract_mask:
.byte 2, 3, 0x80, 0x80, 6, 7, 0x80, 0x80, 10, 11, 0x80, 0x80, 14, 15, 0x80, 0x80
.byte 2, 3, 0x80, 0x80, 6, 7, 0x80, 0x80, 10, 11, 0x80, 0x80, 14, 15, 0x80, 0x80

# Compress imag: same pattern as real compress (bytes are already in low position after extract)
.Limag_compress_mask:
.byte 0, 1, 4, 5, 8, 9, 12, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
.byte 0, 1, 4, 5, 8, 9, 12, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80

# exp(-x) polynomial constants (same as silu.s)
.align 32
.Llog2e:     .float 1.4426950408889634, 1.4426950408889634, 1.4426950408889634, 1.4426950408889634, 1.4426950408889634, 1.4426950408889634, 1.4426950408889634, 1.4426950408889634
.Lc0:        .float 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
.Lc1:        .float 0.69314718, 0.69314718, 0.69314718, 0.69314718, 0.69314718, 0.69314718, 0.69314718, 0.69314718
.Lc2:        .float 0.24022650, 0.24022650, 0.24022650, 0.24022650, 0.24022650, 0.24022650, 0.24022650, 0.24022650
.Lc3:        .float 0.05550411, 0.05550411, 0.05550411, 0.05550411, 0.05550411, 0.05550411, 0.05550411, 0.05550411
.Lc4:        .float 0.00961812, 0.00961812, 0.00961812, 0.00961812, 0.00961812, 0.00961812, 0.00961812, 0.00961812
.Li127:      .long 127, 127, 127, 127, 127, 127, 127, 127
.Lneg_zero:  .float -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0
.Ltwo:       .float 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0
.Lclamp_lo:  .float -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0
.Lclamp_hi:  .float 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0
