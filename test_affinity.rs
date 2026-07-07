use core_affinity;
use std::thread;

fn main() {
    let core_ids = core_affinity::get_core_ids().unwrap();
    println!("Available cores: {:?}", core_ids);
    
    // Pin main to 0
    let res = core_affinity::set_for_current(core_ids[0]);
    println!("Main pinned to 0: {}", res);
    
    // Spawn thread and pin to 1
    let t = thread::spawn(move || {
        let res = core_affinity::set_for_current(core_ids[1]);
        println!("Child pinned to 1: {}", res);
    });
    
    t.join().unwrap();
}
