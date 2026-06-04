.section .text
.global pext_unpack_ternary

# System V AMD64 ABI:
# rdi: packed (u64 containing 32 ternary weights, 2 bits each)
# rsi: out (pointer to 32 bytes where the unpacked -1, 0, 1 will be stored)

pext_unpack_ternary:
    push %rbp
    mov %rsp, %rbp
    push %r12

    # FairyFuse BMI2 decoding:
    # Instead of shifting and masking in a loop, we can use PEXT to extract all 
    # the low bits into one register, and all the high bits into another.
    
    mov $0x5555555555555555, %r8  # Mask for Low bits (0101...)
    mov $0xAAAAAAAAAAAAAAAA, %r9  # Mask for High bits (1010...)
    
    # Extract Low bits (Bit 0 of each 2-bit weight)
    pext %r8, %rdi, %rcx  # rcx contains 32 packed low bits
    
    # Extract High bits (Bit 1 of each 2-bit weight)
    pext %r9, %rdi, %rdx  # rdx contains 32 packed high bits
    
    # Now we have 32 weights unpacked into bits.
    # We need to turn them into byte values (-1, 0, 1) and store in *rsi.
    # We can do this efficiently using BMI2 PDEP or just standard byte expansion.
    # For now, we expand them using AVX2.
    
    vmovq %rcx, %xmm0
    vmovq %rdx, %xmm1
    
    # Interleave to get bytes (this is a simplified placeholder for the full FairyFuse logic)
    # The actual implementation of FairyFuse uses PDEP to spread the bits to bytes.
    # Let's use PDEP to spread the 32 bits into 32 bytes (which is 4 u64s).
    
    mov $0x0101010101010101, %r10 # Mask to place 1 bit per byte
    
    # Process first 8 bits (weights 0-7)
    mov %rcx, %r11
    pdep %r10, %r11, %r11 # r11 has low bits for weights 0-7, spaced by 8 bits
    mov %rdx, %r12
    pdep %r10, %r12, %r12 # r12 has high bits for weights 0-7, spaced by 8 bits
    
    # Calculate value: LowBit - HighBit (if weight is 10, value is 0 - 1 = -1)
    # Since 2 in binary is 10, high bit is 1, low bit is 0 -> 0 - 1 = -1 (0xFF in i8)
    sub %r12, %r11
    mov %r11, 0(%rsi)
    
    # Process next 8 bits (weights 8-15)
    shr $8, %rcx
    shr $8, %rdx
    mov %rcx, %r11
    pdep %r10, %r11, %r11
    mov %rdx, %r12
    pdep %r10, %r12, %r12
    sub %r12, %r11
    mov %r11, 8(%rsi)
    
    # Process next 8 bits (weights 16-23)
    shr $8, %rcx
    shr $8, %rdx
    mov %rcx, %r11
    pdep %r10, %r11, %r11
    mov %rdx, %r12
    pdep %r10, %r12, %r12
    sub %r12, %r11
    mov %r11, 16(%rsi)
    
    # Process final 8 bits (weights 24-31)
    shr $8, %rcx
    shr $8, %rdx
    mov %rcx, %r11
    pdep %r10, %r11, %r11
    mov %rdx, %r12
    pdep %r10, %r12, %r12
    sub %r12, %r11
    mov %r11, 24(%rsi)

    vzeroupper
    pop %r12
    pop %rbp
    ret
