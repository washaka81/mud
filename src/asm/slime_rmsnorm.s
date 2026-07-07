.section .text
.global slime_rmsnorm_i8_avx2

# System V AMD64 ABI:
# rdi: regs (SlimeRegister*)
# rsi: weights (float*)
# rdx: out_i8 (int8_t*)
# rcx: hidden (size_t)
# xmm0: eps (float)
#
# Returns act_scale (peak / 127.0) in xmm0

slime_rmsnorm_i8_avx2:
    push %rbp
    mov %rsp, %rbp
    push %rbx
    push %r12
    push %r13
    push %r14
    push %r15
    
    # Save arguments
    mov %rdi, %r12  # regs
    mov %rsi, %r13  # weights
    mov %rdx, %r14  # out_i8
    mov %rcx, %r15  # hidden
    
    # ymm1 = sum_sq
    vxorps %ymm1, %ymm1, %ymm1
    
    # loop 1: sum_sq
    mov %r15, %rax
    shr $3, %rax    # hidden / 8 (assuming hidden is multiple of 8)
    mov %r12, %r8   # ptr = regs

.loop1:
    test %rax, %rax
    jz .done1
    
    # regs is an array of 8-byte structs.
    # We want to load 8 f32s from 8 structs (64 bytes total).
    # We can load two 32-byte chunks, and use shufps to extract the f32s.
    vmovups 0(%r8), %ymm2   # [f0, pad, f1, pad, f2, pad, f3, pad]
    vmovups 32(%r8), %ymm3  # [f4, pad, f5, pad, f6, pad, f7, pad]
    
    # ymm2 and ymm3 contain 32-bit floats interlaced with 32-bit junk.
    # We can use vshufps to pack them.
    vshufps $0x88, %ymm2, %ymm2, %ymm4  # [f0, f1, f0, f1, f2, f3, f2, f3] -> actually, it's easier to use vgather or just pshufd.
    # Actually, let's just use vmovshdup or vpermilps, then vblendps.
    # Even simpler:
    # 0(%r8)  = f0
    # 8(%r8)  = f1
    # 16(%r8) = f2
    # 24(%r8) = f3
    # 32(%r8) = f4
    # 40(%r8) = f5
    # 48(%r8) = f6
    # 56(%r8) = f7
    
    # Load individually into xmm and unpack?
    # Or load 64-bit pairs.
    # We can load 256-bit ymm.
    # In ymm2: Dwords 0, 2, 4, 6 are the floats.
    # In ymm3: Dwords 0, 2, 4, 6 are the floats.
    
    # Pack ymm2 and ymm3 into ymm4.
    # ymm2 = [A, junk, B, junk, C, junk, D, junk]
    # vpshufd $0x08, %ymm2, %ymm4 -> [A, A, B, A, C, C, D, C] (not quite)
    # The standard way to pack even dwords from two ymm registers:
    vshufps $0x88, %ymm3, %ymm2, %ymm4 # ymm2(0,2), ymm3(0,2), ymm2(4,6), ymm3(4,6)
    # ymm4 = [A, B, E, F, C, D, G, H]
    # Now we need to reorder to [A, B, C, D, E, F, G, H]
    vpermq $0xD8, %ymm4, %ymm4
    # Wait, vpermq operates on 64-bit blocks.
    # [A, B] (0), [E, F] (1), [C, D] (2), [G, H] (3)
    # 0xD8 = 11_01_10_00 = 3, 1, 2, 0. So [A,B], [C,D], [E,F], [G,H]! Perfect.
    
    vmulps %ymm4, %ymm4, %ymm4 # x^2
    vaddps %ymm4, %ymm1, %ymm1 # acc += x^2
    
    add $64, %r8
    dec %rax
    jmp .loop1

.done1:
    # ymm1 contains partial sum_sq. Reduce to xmm1.
    vextractf128 $1, %ymm1, %xmm2
    vaddps %xmm2, %xmm1, %xmm1
    vshufps $0xEE, %xmm1, %xmm1, %xmm2
    vaddps %xmm2, %xmm1, %xmm1
    vshufps $0x11, %xmm1, %xmm1, %xmm2
    vaddps %xmm2, %xmm1, %xmm1
    
    # xmm1[0] = sum_sq.
    # rms_inv = 1.0 / sqrt(sum_sq / hidden + eps)
    vcvtsi2ss %r15, %xmm3, %xmm3 # hidden
    vdivss %xmm3, %xmm1, %xmm1   # mean = sum_sq / hidden
    vaddss %xmm0, %xmm1, %xmm1   # mean + eps
    vsqrtss %xmm1, %xmm1, %xmm1  # sqrt(mean + eps)
    mov $0x3f800000, %r11d
    vmovd %r11d, %xmm2            # 1.0
    vdivss %xmm1, %xmm2, %xmm2   # xmm2 = rms_inv
    # Broadcast rms_inv to ymm2
    vbroadcastss %xmm2, %ymm2
    
    # Pass 2: compute peak
    vxorps %ymm3, %ymm3, %ymm3   # ymm3 = peak_ymm (all 0s)
    # ymm15 = absolute value mask (0x7FFFFFFF)
    mov $0x7FFFFFFF, %r11d
    vmovd %r11d, %xmm15
    vpbroadcastd %xmm15, %ymm15
    
    mov %r15, %rax
    shr $3, %rax
    mov %r12, %r8   # regs
    mov %r13, %r9   # weights

.loop2:
    test %rax, %rax
    jz .done2
    
    # Load 8 regs
    vmovups 0(%r8), %ymm4
    vmovups 32(%r8), %ymm5
    vshufps $0x88, %ymm5, %ymm4, %ymm6
    vpermq $0xD8, %ymm6, %ymm6
    
    # Load 8 weights
    vmovups 0(%r9), %ymm7
    
    # xn = regs * rms_inv * weights
    vmulps %ymm2, %ymm6, %ymm6
    vmulps %ymm7, %ymm6, %ymm6
    
    # abs(xn)
    vandps %ymm15, %ymm6, %ymm6
    
    # peak_ymm = max(peak_ymm, abs(xn))
    vmaxps %ymm6, %ymm3, %ymm3
    
    add $64, %r8
    add $32, %r9
    dec %rax
    jmp .loop2

.done2:
    # Reduce ymm3 to xmm3 (max peak)
    vextractf128 $1, %ymm3, %xmm4
    vmaxps %xmm4, %xmm3, %xmm3
    vshufps $0xEE, %xmm3, %xmm3, %xmm4
    vmaxps %xmm4, %xmm3, %xmm3
    vshufps $0x11, %xmm3, %xmm3, %xmm4
    vmaxps %xmm4, %xmm3, %xmm3
    # xmm3[0] = peak_xn
    
    # peak_xn = max(peak_xn, 1e-8)
    mov $0x33d6bf95, %r11d # 1e-8
    vmovd %r11d, %xmm4
    vmaxss %xmm4, %xmm3, %xmm3
    
    # inv_peak = 127.0 / peak_xn
    mov $0x42fe0000, %r11d # 127.0
    vmovd %r11d, %xmm4
    vdivss %xmm3, %xmm4, %xmm4 # xmm4 = inv_peak
    # Broadcast (rms_inv * inv_peak) to ymm4
    vmulss %xmm2, %xmm4, %xmm5 # xmm5 = rms_inv * inv_peak
    vbroadcastss %xmm5, %ymm4
    
    # Save act_scale = peak / 127.0 in xmm0 for return
    vmovaps %xmm3, %xmm0
    mov $0x42fe0000, %r11d # 127.0
    vmovd %r11d, %xmm1
    vdivss %xmm1, %xmm0, %xmm0
    
    # Pass 3: quantize to i8
    mov %r15, %rax
    shr $3, %rax
    mov %r12, %r8   # regs
    mov %r13, %r9   # weights
    mov %r14, %r10  # out_i8
    
    # For clamping
    mov $0x42fe0000, %r11d # 127.0
    vmovd %r11d, %xmm12
    vbroadcastss %xmm12, %ymm12
    mov $0xc2fe0000, %r11d # -127.0
    vmovd %r11d, %xmm13
    vbroadcastss %xmm13, %ymm13

.loop3:
    test %rax, %rax
    jz .done3
    
    # Load 8 regs
    vmovups 0(%r8), %ymm5
    vmovups 32(%r8), %ymm6
    vshufps $0x88, %ymm6, %ymm5, %ymm7
    vpermq $0xD8, %ymm7, %ymm7
    
    # Load 8 weights
    vmovups 0(%r9), %ymm8
    
    # xn_scaled = regs * weights * (rms_inv * inv_peak)
    vmulps %ymm4, %ymm7, %ymm7
    vmulps %ymm8, %ymm7, %ymm7
    
    # clamp(-127.0, 127.0)
    vmaxps %ymm13, %ymm7, %ymm7
    vminps %ymm12, %ymm7, %ymm7
    
    # Convert to int32 (truncation / nearest)
    vcvttps2dq %ymm7, %ymm7
    
    # Now we have 8 int32s in ymm7. We need to pack to 8 int8s.
    # vpmovdb ymm7 -> xmm7 (Requires AVX512, we can't use it!)
    # We must pack manually using AVX2.
    
    # Pack dword to word: vpackssdw
    # The source is a single YMM register (ymm7).
    # vpackssdw ymm7, ymm7, ymm8 -> packs words. But it operates on 128-bit lanes.
    # ymm7 = [A,B,C,D | E,F,G,H] (dwords)
    # vpackssdw %ymm7, %ymm7, %ymm8 -> [A,B,C,D, A,B,C,D | E,F,G,H, E,F,G,H] (words)
    
    # Let's extract lanes:
    vextracti128 $1, %ymm7, %xmm8
    # xmm7 = [A,B,C,D]
    # xmm8 = [E,F,G,H]
    
    # pack dword to word
    vpackssdw %xmm8, %xmm7, %xmm7  # [A,B,C,D, E,F,G,H] (words)
    
    # pack word to byte
    # We just pack with itself (or zero)
    vpxor %xmm8, %xmm8, %xmm8
    vpacksswb %xmm8, %xmm7, %xmm7  # [A,B,C,D,E,F,G,H, 0...] (bytes)
    
    # Store 8 bytes
    vmovq %xmm7, 0(%r10)
    
    add $64, %r8
    add $32, %r9
    add $8, %r10
    dec %rax
    jmp .loop3

.done3:
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %rbx
    vzeroupper
    pop %rbp
    ret

