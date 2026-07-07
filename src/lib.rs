pub mod asm;
pub mod gguf;
pub mod hardware;
pub mod model;
pub mod mud;
pub mod vulkan;

pub static SHOULD_INTERRUPT_GEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
