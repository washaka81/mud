use core_affinity;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

type Job = Box<dyn FnOnce() + Send + 'static>;

pub struct PCorePool {
    workers: Vec<Worker>,
    slots: Arc<Vec<AtomicPtr<Job>>>,
    pending_tasks: Arc<AtomicUsize>,
    terminate: Arc<AtomicBool>,
    num_threads: usize,
}

struct Worker {
    thread: Option<thread::JoinHandle<()>>,
}

impl PCorePool {
    pub fn new(num_threads: usize) -> PCorePool {
        let mut slots_vec: Vec<AtomicPtr<Job>> = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            slots_vec.push(AtomicPtr::new(std::ptr::null_mut()));
        }
        let slots = Arc::new(slots_vec);
        let pending_tasks = Arc::new(AtomicUsize::new(0));
        let terminate = Arc::new(AtomicBool::new(false));

        let mut workers = Vec::with_capacity(num_threads);
        let core_ids = core_affinity::get_core_ids().unwrap_or_default();

        for id in 0..num_threads {
            let pending_clone = Arc::clone(&pending_tasks);
            let terminate_clone = Arc::clone(&terminate);
            let slots_clone = Arc::clone(&slots);
            let slot_idx = id;

            // Try to assign to P-cores
            let core_id = if id < core_ids.len() {
                Some(core_ids[id])
            } else {
                None
            };

            let thread = thread::spawn(move || {
                if let Some(cid) = core_id {
                    core_affinity::set_for_current(cid);
                }

                loop {
                    if terminate_clone.load(Ordering::SeqCst) {
                        break;
                    }

                    let slot = &slots_clone[slot_idx];
                    let ptr = slot.load(Ordering::Acquire);

                    if !ptr.is_null() {
                        let job = unsafe { Box::from_raw(ptr) };
                        job();

                        pending_clone.fetch_sub(1, Ordering::Release);
                        slot.store(std::ptr::null_mut(), Ordering::Release);
                    } else {
                        // Yield so Drop/join and other cores aren't starved by pure spin
                        std::thread::yield_now();
                    }
                }
            });

            workers.push(Worker {
                thread: Some(thread),
            });
        }

        PCorePool {
            workers,
            slots,
            pending_tasks,
            terminate,
            num_threads,
        }
    }

    /// Worker count (for row splits in GEMV / pack loops).
    #[inline]
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f) as Job;
        let job_ptr = Box::into_raw(Box::new(job));

        self.pending_tasks.fetch_add(1, Ordering::Release);

        loop {
            for slot in self.slots.iter() {
                if slot.load(Ordering::Acquire).is_null()
                    && slot
                        .compare_exchange(
                            std::ptr::null_mut(),
                            job_ptr,
                            Ordering::Release,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                {
                    return;
                }
            }
            std::hint::spin_loop();
        }
    }

    pub fn wait_all(&self) {
        while self.pending_tasks.load(Ordering::Acquire) > 0 {
            std::hint::spin_loop();
        }
    }
}

impl Drop for PCorePool {
    fn drop(&mut self) {
        self.terminate.store(true, Ordering::SeqCst);
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                let _ = thread.join();
            }
        }
    }
}

use std::sync::OnceLock;
pub static GLOBAL_PCORE_POOL: OnceLock<PCorePool> = OnceLock::new();

/// Global pool sized by [`crate::mud::constants::default_pcore_threads`] (L-07).
/// Override with env `MUD_PCORE_THREADS`.
///
/// Captures HW core count **before** constructing workers so a later main-thread
/// pin cannot shrink sizing via affinity-masked `get_core_ids`.
pub fn get_pool() -> &'static PCorePool {
    GLOBAL_PCORE_POOL.get_or_init(|| {
        let _ = crate::mud::constants::capture_hw_pcore_threads();
        let n = crate::mud::constants::default_pcore_threads();
        PCorePool::new(n)
    })
}

/// Thread count of the global pool (initializes it if needed).
#[inline]
pub fn global_pool_threads() -> usize {
    get_pool().num_threads()
}
