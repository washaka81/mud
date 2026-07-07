use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("=== MUD OS FORGE ===");
    println!("Initiating Phase ∞ (SING-01): Autonomous Assembly-Native MUD Kernel generation...");

    let os_dir = Path::new("mud_os_workspace");
    if !os_dir.exists() {
        fs::create_dir(os_dir).expect("Failed to create mud_os_workspace directory");
    }

    // 1. Multiboot Header and Boot Code (GNU Assembly)
    let boot_s = r#"
.set ALIGN,    1<<0             
.set MEMINFO,  1<<1             
.set FLAGS,    ALIGN | MEMINFO  
.set MAGIC,    0x1BADB002       
.set CHECKSUM, -(MAGIC + FLAGS) 

.section .multiboot
.align 4
.long MAGIC
.long FLAGS
.long CHECKSUM

.section .bss
.align 16
stack_bottom:
.skip 16384
stack_top:

.section .text
.global _start
.type _start, @function
_start:
    mov $stack_top, %esp
    call kernel_main

1:  hlt
    jmp 1b

.size _start, . - _start
"#;
    fs::write(os_dir.join("boot.s"), boot_s).unwrap();

    // 2. Kernel Main (GNU Assembly)
    let kernel_s = r#"
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
"#;
    fs::write(os_dir.join("kernel.s"), kernel_s).unwrap();

    // 3. Linker Script
    let linker_ld = r#"
ENTRY(_start)

SECTIONS
{
    . = 1M;
    
    .text BLOCK(4K) : ALIGN(4K)
    {
        *(.multiboot)
        *(.text)
    }
    
    .rodata BLOCK(4K) : ALIGN(4K)
    {
        *(.rodata)
    }
    
    .data BLOCK(4K) : ALIGN(4K)
    {
        *(.data)
    }
    
    .bss BLOCK(4K) : ALIGN(4K)
    {
        *(COMMON)
        *(.bss)
    }
}
"#;
    fs::write(os_dir.join("linker.ld"), linker_ld).unwrap();

    println!("Compiling bare-metal boot sequence (boot.s)...");
    let status = Command::new("as")
        .args(["--32", "boot.s", "-o", "boot.o"])
        .current_dir(os_dir)
        .status();

    if status.is_err() || !status.unwrap().success() {
        eprintln!("Failed to compile boot.s. Ensure GNU binutils ('as') is installed for 32-bit.");
        return;
    }

    println!("Compiling bare-metal MUD kernel (kernel.s)...");
    let status = Command::new("as")
        .args(["--32", "kernel.s", "-o", "kernel.o"])
        .current_dir(os_dir)
        .status();

    if status.is_err() || !status.unwrap().success() {
        eprintln!("Failed to compile kernel.s");
        return;
    }

    println!("Linking autonomous MUD Kernel image...");
    let status = Command::new("ld")
        .args([
            "-m",
            "elf_i386",
            "-T",
            "linker.ld",
            "-o",
            "mud_kernel.elf",
            "boot.o",
            "kernel.o",
        ])
        .current_dir(os_dir)
        .status();

    if status.is_err() || !status.unwrap().success() {
        eprintln!("Failed to link mud_kernel.elf");
        return;
    }

    println!(
        "\n[SUCCESS] MUD-Kernel Assembly-Native generated at: mud_os_workspace/mud_kernel.elf"
    );
    println!("The kernel is fully Multiboot1 compliant.");
    println!("To run it on QEMU, install qemu-system-x86_64 and run:");
    println!("    qemu-system-i386 -kernel mud_os_workspace/mud_kernel.elf");
}
