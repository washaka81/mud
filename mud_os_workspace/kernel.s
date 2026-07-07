
.section .data
vga_start: .long 0xB8000
hello_msg: .asciz "MUD-OS v1.0 Kernel (SING-01: Zero Abstractions, Max Efficiency)"

.section .text
.global kernel_main
.type kernel_main, @function
kernel_main:
    mov vga_start, %edi
    mov $hello_msg, %esi
    mov $0x0F, %ah  /* White on black */

.print_loop:
    lodsb
    test %al, %al
    jz .done
    mov %al, (%edi)
    mov %ah, 1(%edi)
    add $2, %edi
    jmp .print_loop

.done:
    ret
