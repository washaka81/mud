.section .text
.global pext_unpack_ternary

# System V AMD64 ABI:
# rdi: packed (u64 containing 32 ternary weights, 2 bits each)
# rsi: out (pointer to 32 bytes where the unpacked -1, 0, 1 will be stored)

pext_unpack_ternary:
    push %rbp
    mov %rsp, %rbp
    push %r12

    mov $0x5555555555555555, %r8  # Mask for Low bits (0101...)
    mov $0xAAAAAAAAAAAAAAAA, %r9  # Mask for High bits (1010...)
    mov $0x0101010101010101, %r10 # Spread mask: 1 bit per byte

    pext %r8, %rdi, %rcx   # rcx = 32 packed low bits
    pext %r9, %rdi, %rdx   # rdx = 32 packed high bits

    # Group 1 (weights 0-7)
    mov %rcx, %r11
    pdep %r10, %r11, %r11  # r11 = 8 bytes of low bits (0 or 1 per byte)
    mov %rdx, %r12
    pdep %r10, %r12, %r12  # r12 = 8 bytes of high bits (0 or 1 per byte)
    vmovq %r11, %xmm0
    vmovq %r12, %xmm1
    vpsubb %xmm1, %xmm0, %xmm2  # xmm2 = low - high (byte-wise, no borrow)
    vmovq %xmm2, 0(%rsi)

    # Group 2 (weights 8-15)
    shr $8, %rcx
    shr $8, %rdx
    mov %ecx, %r11d
    pdep %r10, %r11, %r11
    mov %edx, %r12d
    pdep %r10, %r12, %r12
    vmovq %r11, %xmm0
    vmovq %r12, %xmm1
    vpsubb %xmm1, %xmm0, %xmm2
    vmovq %xmm2, 8(%rsi)

    # Group 3 (weights 16-23)
    shr $8, %ecx
    shr $8, %edx
    mov %ecx, %r11d
    pdep %r10, %r11, %r11
    mov %edx, %r12d
    pdep %r10, %r12, %r12
    vmovq %r11, %xmm0
    vmovq %r12, %xmm1
    vpsubb %xmm1, %xmm0, %xmm2
    vmovq %xmm2, 16(%rsi)

    # Group 4 (weights 24-31)
    shr $8, %ecx
    shr $8, %edx
    mov %ecx, %r11d
    pdep %r10, %r11, %r11
    mov %edx, %r12d
    pdep %r10, %r12, %r12
    vmovq %r11, %xmm0
    vmovq %r12, %xmm1
    vpsubb %xmm1, %xmm0, %xmm2
    vmovq %xmm2, 24(%rsi)

    vzeroupper
    pop %r12
    pop %rbp
    ret
