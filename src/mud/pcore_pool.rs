use std::sync::Arc;
use parking_lot::{Mutex, Condvar};
use std::thread;
use core_affinity;

type Job = Box<dyn FnOnce() + Send + 'static>;

enum Message {
    NewJob(Job),
    Terminate,
}

struct SharedState {
    jobs: Vec<Message>,
    pending_tasks: usize,
}

pub struct PCorePool {
    workers: Vec<Worker>,
    state: Arc<(Mutex<SharedState>, Condvar)>,
    sync_cv: Arc<Condvar>,
}

struct Worker {
    thread: Option<thread::JoinHandle<()>>,
}

impl PCorePool {
    pub fn new(num_threads: usize) -> PCorePool {
        let state = Arc::new((Mutex::new(SharedState { jobs: Vec::new(), pending_tasks: 0 }), Condvar::new()));
        let sync_cv = Arc::new(Condvar::new());
        let mut workers = Vec::with_capacity(num_threads);

        let core_ids = core_affinity::get_core_ids().unwrap_or_default();

        for id in 0..num_threads {
            let state_clone = Arc::clone(&state);
            let sync_cv_clone = Arc::clone(&sync_cv);
            
            // Try to assign to P-cores (cores 0 to 3)
            let core_id = if id < core_ids.len() { Some(core_ids[id]) } else { None };

            let thread = thread::spawn(move || {
                if let Some(cid) = core_id {
                    core_affinity::set_for_current(cid);
                }
                
                loop {
                    let job = {
                        let (lock, cvar) = &*state_clone;
                        let mut state = lock.lock();
                        while state.jobs.is_empty() {
                            cvar.wait(&mut state);
                        }
                        state.jobs.pop().unwrap()
                    };

                    match job {
                        Message::NewJob(job) => {
                            job();
                            let (lock, _) = &*state_clone;
                            let mut state = lock.lock();
                            state.pending_tasks -= 1;
                            if state.pending_tasks == 0 {
                                sync_cv_clone.notify_all();
                            }
                        }
                        Message::Terminate => break,
                    }
                }
            });

            workers.push(Worker { thread: Some(thread) });
        }

        PCorePool { workers, state, sync_cv }
    }

    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let (lock, cvar) = &*self.state;
        let mut state = lock.lock();
        state.pending_tasks += 1;
        state.jobs.push(Message::NewJob(Box::new(f)));
        cvar.notify_one();
    }
    
    pub fn wait_all(&self) {
        let (lock, _) = &*self.state;
        let mut state = lock.lock();
        while state.pending_tasks > 0 {
            self.sync_cv.wait(&mut state);
        }
    }
}

impl Drop for PCorePool {
    fn drop(&mut self) {
        {
            let (lock, cvar) = &*self.state;
            let mut state = lock.lock();
            for _ in &self.workers {
                state.jobs.push(Message::Terminate);
            }
            cvar.notify_all();
        }
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                thread.join().unwrap();
            }
        }
    }
}

use std::sync::OnceLock;
pub static GLOBAL_PCORE_POOL: OnceLock<PCorePool> = OnceLock::new();
pub fn get_pool() -> &'static PCorePool {
    GLOBAL_PCORE_POOL.get_or_init(|| PCorePool::new(8))
}
