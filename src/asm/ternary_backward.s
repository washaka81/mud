	.file	"ternary_backward.c"
	.text
	.p2align 4
	.globl	ternary_gemv_backward_avx2
	.type	ternary_gemv_backward_avx2, @function
ternary_gemv_backward_avx2:
.LFB0:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	andq	$-32, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	%rdx, -24(%rsp)
	movq	16(%rbp), %r12
	movq	24(%rbp), %rdx
	testq	%r12, %r12
	je	.L80
	movq	%rdi, %rbx
	movq	%r8, %r14
	movq	%rcx, %r15
	movq	%rdi, %rcx
	leaq	-1(%rdx), %rdi
	movq	%r14, %r11
	movq	%rbx, -32(%rsp)
	movq	%r14, %rax
	movq	%rdi, -16(%rsp)
	movq	%rdx, %rdi
	subq	%rsi, %r11
	movq	-16(%rsp), %r13
	shrq	$3, %rdi
	movq	%r15, -40(%rsp)
	subq	$4, %r11
	xorl	%r10d, %r10d
	movq	%rdi, -8(%rsp)
	leaq	(%rbx,%r12,4), %r12
	salq	$5, %rdi
	vmovss	.LC0(%rip), %xmm4
	movq	%r9, -48(%rsp)
	leaq	0(,%rdx,4), %r8
	jmp	.L11
	.p2align 4,,10
	.p2align 3
.L5:
	addq	$4, %rcx
	addq	%rdx, %r10
	addq	%r8, %rax
	addq	%r8, %r11
	cmpq	%rcx, %r12
	je	.L83
.L11:
	vmovss	(%rcx), %xmm1
	vcomiss	%xmm1, %xmm4
	vbroadcastss	%xmm1, %ymm3
	jbe	.L3
	vcomiss	.LC1(%rip), %xmm1
	ja	.L5
.L3:
	testq	%rdx, %rdx
	je	.L5
	cmpq	$2, %r13
	jbe	.L41
	cmpq	$24, %r11
	jbe	.L41
	cmpq	$6, %r13
	jbe	.L42
	vmovaps	%ymm3, %ymm2
	xorl	%r9d, %r9d
	.p2align 5
	.p2align 4
	.p2align 3
.L8:
	vmulps	(%rsi,%r9), %ymm2, %ymm0
	vaddps	(%rax,%r9), %ymm0, %ymm0
	vmovups	%ymm0, (%rax,%r9)
	addq	$32, %r9
	cmpq	%r9, %rdi
	jne	.L8
	movq	-8(%rsp), %rbx
	leaq	0(,%rbx,8), %r9
	cmpq	%r9, %rdx
	je	.L5
	movq	%rdx, %rbx
	subq	%r9, %rbx
	leaq	-1(%rbx), %r15
	cmpq	$2, %r15
	jbe	.L9
.L7:
	leaq	(%r10,%r9), %r15
	vmulps	(%rsi,%r9,4), %xmm3, %xmm0
	vaddps	(%r14,%r15,4), %xmm0, %xmm0
	vmovups	%xmm0, (%r14,%r15,4)
	movq	%rbx, %r15
	andq	$-4, %r15
	andl	$3, %ebx
	je	.L5
	addq	%r15, %r9
.L9:
	leaq	(%r10,%r9), %rbx
	vmulss	(%rsi,%r9,4), %xmm1, %xmm0
	vaddss	(%r14,%rbx,4), %xmm0, %xmm0
	vmovss	%xmm0, (%r14,%rbx,4)
	leaq	1(%r9), %rbx
	cmpq	%rdx, %rbx
	jnb	.L5
	vmulss	4(%rsi,%r9,4), %xmm1, %xmm0
	addq	%r10, %rbx
	vaddss	(%r14,%rbx,4), %xmm0, %xmm0
	vmovss	%xmm0, (%r14,%rbx,4)
	leaq	2(%r9), %rbx
	cmpq	%rdx, %rbx
	jnb	.L5
	vmulss	8(%rsi,%r9,4), %xmm1, %xmm1
	addq	%r10, %rbx
	vaddss	(%r14,%rbx,4), %xmm1, %xmm1
	vmovss	%xmm1, (%r14,%rbx,4)
	jmp	.L5
.L83:
	shrq	$3, %rdx
	movq	-48(%rsp), %r13
	movq	16(%rbp), %r12
	xorl	%r9d, %r9d
	movq	%rdx, %rdi
	movq	-32(%rsp), %rbx
	movq	-40(%rsp), %r15
	xorl	%r8d, %r8d
	salq	$5, %rdi
	movq	-24(%rsp), %r10
	addq	%r13, %rdi
.L39:
	vmovss	(%rbx,%r8,4), %xmm0
	vcomiss	%xmm0, %xmm4
	jbe	.L12
.L37:
	vcomiss	.LC1(%rip), %xmm0
	jbe	.L12
.L14:
	addq	$1, %r8
	addq	%rdx, %r9
	cmpq	%r8, %r12
	jne	.L39
.L79:
	vzeroupper
.L80:
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L12:
	.cfi_restore_state
	testq	%rdx, %rdx
	je	.L15
	vxorps	%xmm3, %xmm3, %xmm3
.L38:
	vmulss	(%r15,%r8,4), %xmm0, %xmm0
	leaq	(%r10,%r9,4), %rsi
	movq	%r13, %rax
	vxorps	.LC3(%rip), %xmm0, %xmm1
	.p2align 4
	.p2align 3
.L32:
	movl	(%rsi), %ecx
	vmovaps	%xmm0, %xmm2
	movl	%ecx, %r11d
	andl	$15, %r11d
	cmpl	$1, %r11d
	je	.L16
	vmovaps	%xmm1, %xmm2
	cmpl	$15, %r11d
	je	.L16
	vmulss	%xmm3, %xmm0, %xmm2
.L16:
	vaddss	(%rax), %xmm2, %xmm2
	movl	%ecx, %r11d
	shrl	$4, %r11d
	andl	$15, %r11d
	vmovss	%xmm2, (%rax)
	vmovaps	%xmm0, %xmm2
	cmpl	$1, %r11d
	je	.L18
	vmovaps	%xmm1, %xmm2
	cmpl	$15, %r11d
	je	.L18
	vmulss	%xmm3, %xmm0, %xmm2
.L18:
	vaddss	4(%rax), %xmm2, %xmm2
	movl	%ecx, %r11d
	shrl	$8, %r11d
	andl	$15, %r11d
	vmovss	%xmm2, 4(%rax)
	vmovaps	%xmm0, %xmm2
	cmpl	$1, %r11d
	je	.L20
	vmovaps	%xmm1, %xmm2
	cmpl	$15, %r11d
	je	.L20
	vmulss	%xmm3, %xmm0, %xmm2
.L20:
	vaddss	8(%rax), %xmm2, %xmm2
	movl	%ecx, %r11d
	shrl	$12, %r11d
	andl	$15, %r11d
	vmovss	%xmm2, 8(%rax)
	vmovaps	%xmm0, %xmm2
	cmpl	$1, %r11d
	je	.L22
	vmovaps	%xmm1, %xmm2
	cmpl	$15, %r11d
	je	.L22
	vmulss	%xmm3, %xmm0, %xmm2
.L22:
	vaddss	12(%rax), %xmm2, %xmm2
	movl	%ecx, %r11d
	shrl	$16, %r11d
	andl	$15, %r11d
	vmovss	%xmm2, 12(%rax)
	vmovaps	%xmm0, %xmm2
	cmpl	$1, %r11d
	je	.L24
	vmovaps	%xmm1, %xmm2
	cmpl	$15, %r11d
	je	.L24
	vmulss	%xmm3, %xmm0, %xmm2
.L24:
	vaddss	16(%rax), %xmm2, %xmm2
	movl	%ecx, %r11d
	shrl	$20, %r11d
	andl	$15, %r11d
	vmovss	%xmm2, 16(%rax)
	vmovaps	%xmm0, %xmm2
	cmpl	$1, %r11d
	je	.L26
	vmovaps	%xmm1, %xmm2
	cmpl	$15, %r11d
	je	.L26
	vmulss	%xmm3, %xmm0, %xmm2
.L26:
	vaddss	20(%rax), %xmm2, %xmm2
	movl	%ecx, %r11d
	shrl	$24, %r11d
	andl	$15, %r11d
	vmovss	%xmm2, 20(%rax)
	vmovaps	%xmm0, %xmm2
	cmpl	$1, %r11d
	je	.L28
	vmovaps	%xmm1, %xmm2
	cmpl	$15, %r11d
	je	.L28
	vmulss	%xmm3, %xmm0, %xmm2
.L28:
	vaddss	24(%rax), %xmm2, %xmm2
	shrl	$28, %ecx
	vmovss	%xmm2, 24(%rax)
	vmovaps	%xmm0, %xmm2
	cmpl	$1, %ecx
	je	.L30
	vmovaps	%xmm1, %xmm2
	cmpl	$15, %ecx
	je	.L30
	vmulss	%xmm3, %xmm0, %xmm2
.L30:
	vaddss	28(%rax), %xmm2, %xmm2
	addq	$32, %rax
	addq	$4, %rsi
	vmovss	%xmm2, -4(%rax)
	cmpq	%rax, %rdi
	jne	.L32
	addq	$1, %r8
	cmpq	%r12, %r8
	je	.L79
	vmovss	(%rbx,%r8,4), %xmm0
	addq	%rdx, %r9
	vcomiss	%xmm0, %xmm4
	ja	.L37
	jmp	.L38
.L84:
	vmovss	(%rbx,%rax,4), %xmm0
	vcomiss	%xmm0, %xmm4
	jbe	.L76
	vcomiss	.LC1(%rip), %xmm0
	ja	.L51
.L36:
	movq	%rax, %r8
.L15:
	leaq	1(%r8), %rax
	cmpq	%r12, %rax
	jne	.L84
	jmp	.L79
	.p2align 4,,10
	.p2align 3
.L41:
	xorl	%r9d, %r9d
	.p2align 5
	.p2align 4
	.p2align 3
.L10:
	vmulss	(%rsi,%r9,4), %xmm1, %xmm0
	vaddss	(%rax,%r9,4), %xmm0, %xmm0
	vmovss	%xmm0, (%rax,%r9,4)
	addq	$1, %r9
	cmpq	%r9, %rdx
	jne	.L10
	jmp	.L5
.L42:
	movq	%rdx, %rbx
	xorl	%r9d, %r9d
	jmp	.L7
.L51:
	movq	%rax, %r8
	jmp	.L14
.L76:
	addq	$2, %r8
	cmpq	%r8, %r12
	je	.L79
	vmovss	(%rbx,%r8,4), %xmm0
	vcomiss	%xmm0, %xmm4
	ja	.L37
	movq	%r8, %rax
	jmp	.L36
	.cfi_endproc
.LFE0:
	.size	ternary_gemv_backward_avx2, .-ternary_gemv_backward_avx2
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC0:
	.long	841731191
	.align 4
.LC1:
	.long	-1305752457
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC3:
	.long	-2147483648
	.long	0
	.long	0
	.long	0
	.ident	"GCC: (GNU) 16.1.1 20260430"
	.section	.note.GNU-stack,"",@progbits
