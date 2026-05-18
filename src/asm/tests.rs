use half::f16;
use crate::asm::*;

#[test]
fn test_basic_gemv_regression() {
    let n = 32;
    let x = vec![1.0f32; n];
    let mut out = vec![0.0f32; 1]; // Only needs one float for the reduced result
    let weights = vec![BlockQ4_0 { d: f16::from_f32(1.0), qs: [0x99; 16] }];
    
    // The kernel adds to *out, so it must be initialized to 0.0
    unsafe {
        q4_0_gemv_asm(n, x.as_ptr(), weights.as_ptr(), out.as_mut_ptr());
    }
    
    println!("Basic GEMV Result: {:?}", out[0]);
    // 0x99 -> both nibbles are 9. 
    // 9 - 8 = 1.0 real value.
    // 32 elements * 1.0 * 1.0 = 32.0
    assert_eq!(out[0], 32.0);
}
