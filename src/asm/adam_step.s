/* adam_step_avx2 — Adam optimizer step (AVX2 + FMA), AT&T syntax
 *
 * void adam_step_avx2(
 *   size_t  n,          rdi
 *   float*  w,          rsi
 *   float*  m,          rdx
 *   float*  v,          rcx
 *   float*  grads,      r8
 *   float   clip_coef,  xmm0
 *   float   wd,         xmm1
 *   float   b1,         xmm2
 *   float   b2,         xmm3
 *   float   lr_bc1,     xmm4
 *   float   inv_bc2,    xmm5
 *   float   eps         xmm6
 * );
 *
 * Polish: AT&T (match rest of tree), prefetch 4 streams, NaN grads → 0.
 */

.section .text
.global adam_step_avx2
.type   adam_step_avx2, @function

adam_step_avx2:
    push %rbp
    mov %rsp, %rbp
    push %rbx
    push %r12

    vbroadcastss %xmm0, %ymm8       # clip_coef
    vbroadcastss %xmm1, %ymm9       # wd
    vbroadcastss %xmm2, %ymm10      # b1
    vbroadcastss %xmm3, %ymm13      # b2
    vbroadcastss %xmm4, %ymm12      # lr_bc1
    vbroadcastss %xmm5, %ymm7       # inv_bc2
    vbroadcastss %xmm6, %ymm15      # eps

    # 1.0 broadcast → (1-b1), (1-b2)
    movl $0x3F800000, %eax
    vmovd %eax, %xmm0
    vbroadcastss %xmm0, %ymm0
    vsubps %ymm10, %ymm0, %ymm11    # 1 - b1
    vsubps %ymm13, %ymm0, %ymm14    # 1 - b2

    xor %r12d, %r12d                # i = 0

.Ladam_loop8:
    lea 8(%r12), %rbx
    cmp %rdi, %rbx
    ja .Ladam_tail

    # Prefetch next cachelines for all four streams
    prefetcht0 256(%r8, %r12, 4)
    prefetcht0 256(%rsi, %r12, 4)
    prefetcht0 256(%rdx, %r12, 4)
    prefetcht0 256(%rcx, %r12, 4)

    vmovups (%r8, %r12, 4), %ymm0   # grads
    vmovups (%rsi, %r12, 4), %ymm1  # w
    vmovups (%rdx, %r12, 4), %ymm2  # m
    vmovups (%rcx, %r12, 4), %ymm3  # v

    # NaN/Inf grads → 0 (exponent all-ones)
    # Mask: where (abs(g) has exp==0xFF) replace with 0
    vmovdqa .exp_mask(%rip), %ymm4
    vpand %ymm4, %ymm0, %ymm5
    vpcmpeqd %ymm4, %ymm5, %ymm5    # 0xFFFFFFFF lanes that are NaN/Inf
    vpandn %ymm0, %ymm5, %ymm0      # clear those lanes

    # g = grads * clip + wd * w
    vmulps %ymm8, %ymm0, %ymm4
    vfmadd231ps %ymm9, %ymm1, %ymm4

    # m = b1*m + (1-b1)*g
    vmulps %ymm10, %ymm2, %ymm2
    vfmadd231ps %ymm11, %ymm4, %ymm2

    # v = b2*v + (1-b2)*g*g
    vmulps %ymm4, %ymm4, %ymm5
    vmulps %ymm13, %ymm3, %ymm3
    vfmadd231ps %ymm14, %ymm5, %ymm3

    # step = lr_bc1 * m / (sqrt(v*inv_bc2) + eps)
    vmulps %ymm7, %ymm3, %ymm5
    vsqrtps %ymm5, %ymm5
    vaddps %ymm15, %ymm5, %ymm5
    vdivps %ymm5, %ymm2, %ymm6
    vmulps %ymm12, %ymm6, %ymm6

    vsubps %ymm6, %ymm1, %ymm1

    vmovups %ymm1, (%rsi, %r12, 4)
    vmovups %ymm2, (%rdx, %r12, 4)
    vmovups %ymm3, (%rcx, %r12, 4)

    add $8, %r12
    jmp .Ladam_loop8

.Ladam_tail:
    # Reload scalars into low xmm of broadcast regs (already low dword ok)
.Ladam_tail_loop:
    cmp %rdi, %r12
    jge .Ladam_done

    vmovss (%r8, %r12, 4), %xmm0
    vmovss (%rsi, %r12, 4), %xmm1
    vmovss (%rdx, %r12, 4), %xmm2
    vmovss (%rcx, %r12, 4), %xmm3

    # scalar NaN/Inf kill
    vmovd %xmm0, %eax
    andl $0x7F800000, %eax
    cmpl $0x7F800000, %eax
    jne 1f
    vxorps %xmm0, %xmm0, %xmm0
1:
    vmulss %xmm8, %xmm0, %xmm4
    vfmadd231ss %xmm9, %xmm1, %xmm4

    vmulss %xmm10, %xmm2, %xmm2
    vfmadd231ss %xmm11, %xmm4, %xmm2

    vmulss %xmm4, %xmm4, %xmm5
    vmulss %xmm13, %xmm3, %xmm3
    vfmadd231ss %xmm14, %xmm5, %xmm3

    vmulss %xmm7, %xmm3, %xmm5
    vsqrtss %xmm5, %xmm5, %xmm5
    vaddss %xmm15, %xmm5, %xmm5
    vdivss %xmm5, %xmm2, %xmm6
    vmulss %xmm12, %xmm6, %xmm6
    vsubss %xmm6, %xmm1, %xmm1

    vmovss %xmm1, (%rsi, %r12, 4)
    vmovss %xmm2, (%rdx, %r12, 4)
    vmovss %xmm3, (%rcx, %r12, 4)

    inc %r12
    jmp .Ladam_tail_loop

.Ladam_done:
    vzeroupper
    pop %r12
    pop %rbx
    pop %rbp
    ret

.section .rodata
.align 32
.exp_mask:
    .long 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000
    .long 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000
