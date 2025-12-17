use std::collections::HashMap;
use std::num::Wrapping;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::sleep;
use std::time::Duration;
use crate::pow::{self, BlockSeed};
use crate::pow::SHARE_TRACKER; // <-- global tracker used by report_block()
use crate::watch;
use crate::Error;
use log::{debug, error, info, warn};
use rand::Rng;
use tokio::sync::mpsc::Sender;
use tokio::task::{self, JoinHandle};
use tokio::time::MissedTickBehavior;
use sysinfo::{System};

use vecno_miner::{PluginManager, WorkerSpec};

type MinerHandler = std::thread::JoinHandle<Result<(), Error>>;

#[cfg(any(target_os = "linux", target_os = "macos"))]
extern "C" fn signal_panic(_signal: nix::libc::c_int) {
    panic!("Forced shutdown");
}
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn register_freeze_handler() {
    let handler = nix::sys::signal::SigHandler::Handler(signal_panic);
    unsafe { nix::sys::signal::signal(nix::sys::signal::Signal::SIGUSR1, handler).unwrap(); }
}
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn trigger_freeze_handler(kill_switch: Arc<AtomicBool>, handle: &MinerHandler) -> std::thread::JoinHandle<()> {
    use std::os::unix::thread::JoinHandleExt;
    let pthread_handle = handle.as_pthread_t();
    std::thread::spawn(move || {
        sleep(Duration::from_millis(1000));
        if kill_switch.load(Ordering::SeqCst) {
            match nix::sys::pthread::pthread_kill(pthread_handle, nix::sys::signal::Signal::SIGUSR1) {
                Ok(()) => info!("Thread killed successfully"),
                Err(e) => error!("Error killing thread: {:?}", e),
            }
        }
    })
}

#[cfg(target_os = "windows")]
struct RawHandle(*mut std::ffi::c_void);
#[cfg(target_os = "windows")]
unsafe impl Send for RawHandle {}
#[cfg(target_os = "windows")]
fn register_freeze_handler() {}
#[cfg(target_os = "windows")]
fn trigger_freeze_handler(kill_switch: Arc<AtomicBool>, handle: &MinerHandler) -> std::thread::JoinHandle<()> {
    use std::os::windows::io::AsRawHandle;
    let raw_handle = RawHandle(handle.as_raw_handle());
    std::thread::spawn(move || unsafe {
        let ensure_full_move = raw_handle;
        sleep(Duration::from_millis(1000));
        if kill_switch.load(Ordering::SeqCst) {
            if kernel32::TerminateThread(ensure_full_move.0, 0) == 0 {
                error!("Failed to terminate Windows thread: {}", std::io::Error::last_os_error());
            } else {
                info!("Windows thread terminated successfully");
            }
        }
    })
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn trigger_freeze_handler(_kill_switch: Arc<AtomicBool>, _handle: &MinerHandler) {
    warn!("Freeze handler not implemented. Frozen threads are ignored");
}
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn register_freeze_handler() {
    warn!("Freeze handler not implemented. Frozen threads are ignored");
}

#[derive(Clone)]
enum WorkerCommand {
    Job(Box<pow::State>),
    Close,
}

#[allow(dead_code)]
pub struct MinerManager {
    handles: Vec<MinerHandler>,
    block_channel: watch::Sender<Option<WorkerCommand>>,
    send_channel: Sender<BlockSeed>,
    logger_handle: JoinHandle<()>,
    is_synced: bool,
    hashes_tried: Arc<AtomicU64>,
    hashes_by_worker: Arc<Mutex<HashMap<String, Arc<AtomicU64>>>>,
    current_state_id: Arc<AtomicUsize>,
    cpu_hashes: Arc<AtomicU64>,
    cpu_count: u16,
}

/* Drop impl – unchanged */
impl Drop for MinerManager {
    fn drop(&mut self) {
        info!("Closing miner");
        self.logger_handle.abort();
        match self.block_channel.send(Some(WorkerCommand::Close)) {
            Ok(_) => debug!("Sent close command to workers"),
            Err(e) => warn!("Failed to send close command: {}", e),
        }
        while !self.handles.is_empty() {
            let handle = self.handles.pop().expect("There should be at least one");
            let kill_switch = Arc::new(AtomicBool::new(true));
            trigger_freeze_handler(kill_switch.clone(), &handle);
            match handle.join() {
                Ok(res) => match res {
                    Ok(()) => debug!("Worker thread closed successfully"),
                    Err(e) => error!("Error when closing Worker: {}", e),
                },
                Err(_) => error!("Worker failed to close gracefully"),
            };
            kill_switch.fetch_and(false, Ordering::SeqCst);
        }
    }
}

pub fn get_num_threads(n_threads: Option<u16>) -> u16 {
    let max_threads = num_cpus::get() as u16;
    let requested_threads = n_threads.unwrap_or(max_threads);
    requested_threads.min(max_threads)
}

const LOG_RATE: Duration = Duration::from_secs(10);

impl MinerManager {
    pub fn new(send_channel: Sender<BlockSeed>, n_threads: Option<u16>, manager: &PluginManager) -> Self {
        register_freeze_handler();
        let hashes_tried = Arc::new(AtomicU64::new(0));
        let cpu_hashes = Arc::new(AtomicU64::new(0));

        let thread_count = get_num_threads(n_threads);

        let mut system = System::new_all();
        system.refresh_all();

        let cpu_info = if let Some(cpu) = system.cpus().first() {
            format!("CPU: {}", cpu.brand())
        } else {
            "CPU information unavailable".to_string()
        };

        info!(
            "Detected {} logical threads, using {} threads. {}",
            num_cpus::get() as u16,
            thread_count,
            cpu_info
        );
        if n_threads.unwrap_or(thread_count) > thread_count {
            warn!(
                "Requested {} threads, but limited to {} due to logical thread count",
                n_threads.unwrap_or(thread_count),
                thread_count
            );
        }

        let hashes_by_worker = Arc::new(Mutex::new(HashMap::<String, Arc<AtomicU64>>::new()));
        let (send, recv) = watch::channel(None);
        let mut handles = Self::launch_cpu_threads(
            send_channel.clone(),
            Arc::clone(&hashes_tried),
            Arc::clone(&cpu_hashes),
            recv.clone(),
            thread_count,
        )
        .collect::<Vec<MinerHandler>>();

        if manager.has_specs() {
            debug!("Found GPU specs, launching GPU threads");
            handles.append(&mut Self::launch_gpu_threads(
                send_channel.clone(),
                Arc::clone(&hashes_tried),
                recv,
                manager,
                hashes_by_worker.clone(),
            ));
        } else {
            warn!("No GPU specs found, skipping GPU thread launch");
        }

        Self {
            handles,
            block_channel: send,
            send_channel,
            logger_handle: task::spawn(Self::log_hashrate(
                Arc::clone(&hashes_tried),
                Arc::clone(&cpu_hashes),
                hashes_by_worker.clone(),
                thread_count,
            )),
            is_synced: true,
            hashes_tried,
            hashes_by_worker,
            current_state_id: Arc::new(AtomicUsize::new(0)),
            cpu_hashes,
            cpu_count: thread_count,
        }
    }

    fn launch_cpu_threads(
        send_channel: Sender<BlockSeed>,
        hashes_tried: Arc<AtomicU64>,
        cpu_hashes: Arc<AtomicU64>,
        work_channel: watch::Receiver<Option<WorkerCommand>>,
        thread_count: u16,
    ) -> impl Iterator<Item = MinerHandler> {
        info!("Launching {} CPU threads", thread_count);
        (0..thread_count).map(move |i| {
            Self::launch_cpu_miner(
                send_channel.clone(),
                work_channel.clone(),
                Arc::clone(&hashes_tried),
                Arc::clone(&cpu_hashes),
                i,
                thread_count,
            )
        })
    }

    fn launch_gpu_threads(
        send_channel: Sender<BlockSeed>,
        hashes_tried: Arc<AtomicU64>,
        work_channel: watch::Receiver<Option<WorkerCommand>>,
        manager: &PluginManager,
        hashes_by_worker: Arc<Mutex<HashMap<String, Arc<AtomicU64>>>>,
    ) -> Vec<MinerHandler> {
        let mut vec = Vec::<MinerHandler>::new();
        let specs = manager.build().unwrap();
        for spec in specs {
            let worker_hashes_tried = Arc::new(AtomicU64::new(0));
            hashes_by_worker
                .lock()
                .unwrap()
                .insert(spec.id(), worker_hashes_tried.clone());
            info!("Launching GPU miner for device: {}.", spec.id());
            vec.push(Self::launch_gpu_miner(
                send_channel.clone(),
                work_channel.clone(),
                Arc::clone(&hashes_tried),
                spec,
                worker_hashes_tried,
            ));
        }
        vec
    }

    pub async fn process_block(&mut self, block: Option<BlockSeed>) -> Result<(), Error> {
        let state = match block {
            Some(b) => {
                self.is_synced = true;
                let id = self.current_state_id.fetch_add(1, Ordering::SeqCst);
                Some(WorkerCommand::Job(Box::new(pow::State::new(id, b)?)))
            }
            None => {
                if !self.is_synced {
                    debug!("Node not synced, skipping block processing");
                    return Ok(());
                }
                self.is_synced = false;
                warn!("Vecnod is not synced, skipping current template");
                None
            }
        };
        debug!("Sending block command to workers: {:?}", state.is_some());
        self.block_channel.send(state).map_err(|e| {
            error!("Failed sending block to threads: {}", e);
            "Failed sending block to threads"
        })?;
        Ok(())
    }

    #[allow(unreachable_code)]
    fn launch_cpu_miner(
        send_channel: Sender<BlockSeed>,
        mut block_channel: watch::Receiver<Option<WorkerCommand>>,
        hashes_tried: Arc<AtomicU64>,
        cpu_hashes: Arc<AtomicU64>,
        thread_index: u16,
        thread_count: u16,
    ) -> MinerHandler {
        let mut nonce = Wrapping(0);
        let mut mask = Wrapping(0);
        let mut fixed = Wrapping(0);
        std::thread::spawn(move || {
            (|| {
                let mut state = None;
                let mut nonce_count = 0;
                let mut current_job_id = 0;
                let nonce_range_size = u64::MAX / (thread_count as u64);
                let random_offset = rand::rng().random::<u64>() % nonce_range_size;
                let nonce_start = (thread_index as u64) * nonce_range_size
                    + (fixed.0 % nonce_range_size)
                    + random_offset;
                nonce = Wrapping(nonce_start);
                debug!(
                    "CPU[{}]: Initial nonce range start: {:016x} for job {}",
                    thread_index, nonce_start, current_job_id
                );

                loop {
                    if state.is_none() {
                        state = match block_channel.wait_for_change() {
                            Ok(cmd) => match cmd {
                                Some(WorkerCommand::Job(s)) => {
                                    mask = Wrapping(s.nonce_mask);
                                    fixed = Wrapping(s.nonce_fixed);
                                    current_job_id = s.id;
                                    nonce_count = 0;
                                    let nonce_start = (thread_index as u64) * nonce_range_size
                                        + (fixed.0 % nonce_range_size)
                                        + random_offset;
                                    nonce = Wrapping(nonce_start);
                                    debug!(
                                        "CPU[{}]: New nonce range start: {:016x} for job {}",
                                        thread_index, nonce_start, current_job_id
                                    );
                                    Some(s)
                                }
                                Some(WorkerCommand::Close) => {
                                    info!("CPU[{}]: Received close command", thread_index);
                                    return Ok(());
                                }
                                None => None,
                            },
                            Err(e) => {
                                error!("CPU[{}]: Channel error: {}", thread_index, e);
                                return Ok(());
                            }
                        };
                    }

                    {
                        let state_ref = match state.as_mut() {
                            Some(s) => s,
                            None => continue,
                        };
                        nonce = (nonce & mask) | fixed;

                        if let Some(block_seed) = state_ref.generate_block_if_pow(nonce.0, &SHARE_TRACKER) {
                            // This logs "Successfully found X shares..." for PartialBlock
                            block_seed.report_block();

                            match send_channel.blocking_send(block_seed.clone()) {
                                Ok(()) => {}
                                Err(e) => error!(
                                    "CPU[{}]: Failed submitting block: {}",
                                    thread_index, e
                                ),
                            }

                            if let BlockSeed::FullBlock(_) = block_seed {
                                state = None;
                            }
                        }

                        nonce += Wrapping(thread_count as u64);
                        nonce_count += 1;
                        hashes_tried.fetch_add(1, Ordering::AcqRel);
                        cpu_hashes.fetch_add(1, Ordering::AcqRel);

                        if nonce.0 > u64::MAX - (thread_count as u64 * 1000) {
                            error!(
                                "CPU[{}]: Nonce exhaustion detected for job {}. Disconnecting to request new extranonce.",
                                thread_index, current_job_id
                            );
                            return Err("Nonce exhaustion detected".into());
                        }

                        if nonce_count % 100_000 == 0 {
                            debug!(
                                "CPU[{}]: Tried {} nonces for job {}, current nonce: {:016x}",
                                thread_index, nonce_count, current_job_id, nonce.0
                            );
                        }
                    }

                    if nonce_count % 128 == 0 {
                        if let Some(new_cmd) = block_channel.get_changed()? {
                            state = match new_cmd {
                                Some(WorkerCommand::Job(s)) => {
                                    mask = Wrapping(s.nonce_mask);
                                    fixed = Wrapping(s.nonce_fixed);
                                    current_job_id = s.id;
                                    nonce_count = 0;
                                    let nonce_start = (thread_index as u64) * nonce_range_size
                                        + (fixed.0 % nonce_range_size)
                                        + random_offset;
                                    nonce = Wrapping(nonce_start);
                                    debug!(
                                        "CPU[{}]: New nonce range start: {:016x} for job {}",
                                        thread_index, nonce_start, current_job_id
                                    );
                                    Some(s)
                                }
                                Some(WorkerCommand::Close) => {
                                    info!("CPU[{}]: Closing thread", thread_index);
                                    return Ok(());
                                }
                                None => None,
                            };
                        }
                    }
                }
                Ok(())
            })()
            .map_err(|e: Error| {
                error!("CPU[{}]: Thread crashed: {}", thread_index, e);
                e
            })
        })
    }

    #[allow(unreachable_code)]
    fn launch_gpu_miner(
        send_channel: Sender<BlockSeed>,
        mut block_channel: watch::Receiver<Option<WorkerCommand>>,
        hashes_tried: Arc<AtomicU64>,
        spec: Box<dyn WorkerSpec>,
        worker_hashes_tried: Arc<AtomicU64>,
    ) -> MinerHandler {
        std::thread::spawn(move || {
            let mut box_ = spec.build();
            let gpu_work = box_.as_mut();
            (|| {
                let mut nonces = vec![0u64; 1];
                let mut state = None;
                let mut nonce_count = 0;
                loop {
                    nonces[0] = 0;
                    if state.is_none() {
                        debug!("GPU {}: Waiting for new state", gpu_work.id());
                        state = match block_channel.wait_for_change() {
                            Ok(cmd) => match cmd {
                                Some(WorkerCommand::Job(s)) => {
                                    nonce_count = 0;
                                    Some(s)
                                }
                                Some(WorkerCommand::Close) => {
                                    info!("GPU {}: Received close command", gpu_work.id());
                                    return Ok(());
                                }
                                None => None,
                            },
                            Err(e) => {
                                error!("GPU {}: Channel error: {}", gpu_work.id(), e);
                                return Ok(());
                            }
                        };
                    }

                    {
                        let state_ref = match state.as_mut() {
                            Some(s) => s,
                            None => continue,
                        };
                        state_ref.load_to_gpu(gpu_work);
                        state_ref.pow_gpu(gpu_work);
                        if let Err(e) = gpu_work.sync() {
                            warn!("GPU {}: CUDA sync failed: {}", gpu_work.id(), e);
                            continue;
                        }
                        if let Err(e) = gpu_work.copy_output_to(&mut nonces) {
                            error!("GPU {}: Failed to copy output: {}", gpu_work.id(), e);
                            continue;
                        }

                        if nonces[0] != 0 {
                            if let Some(block_seed) = state_ref.generate_block_if_pow(nonces[0], &SHARE_TRACKER) {
                                // This logs "Successfully found X shares..." for PartialBlock
                                block_seed.report_block();

                                match send_channel.blocking_send(block_seed.clone()) {
                                    Ok(()) => {}
                                    Err(e) => error!(
                                        "GPU {}: Failed submitting block: {}",
                                        gpu_work.id(),
                                        e
                                    ),
                                }

                                if let BlockSeed::FullBlock(_) = block_seed {
                                    state = None;
                                }
                                nonces[0] = 0;
                            } else {
                                let hash = state_ref.calculate_pow(nonces[0]);
                                warn!(
                                    "GPU {}: Invalid nonce {}, GPU hash: {:x}, Target: {:x}",
                                    gpu_work.id(),
                                    nonces[0],
                                    hash,
                                    state_ref.target
                                );
                            }
                        }

                        let workload = gpu_work.get_workload();
                        nonce_count += workload as u64;
                        debug!(
                            "GPU {}: Adding {} hashes, total nonces tried: {}",
                            gpu_work.id(),
                            workload,
                            nonce_count
                        );
                        hashes_tried.fetch_add(workload as u64, Ordering::AcqRel);
                        worker_hashes_tried.fetch_add(workload as u64, Ordering::AcqRel);

                        if nonce_count > u64::MAX - 100_000 {
                            error!(
                                "GPU {}: Nonce exhaustion detected. Disconnecting to request new extranonce.",
                                gpu_work.id()
                            );
                            return Err("Nonce exhaustion detected".into());
                        }
                    }

                    if let Some(new_cmd) = block_channel.get_changed()? {
                        state = match new_cmd {
                            Some(WorkerCommand::Job(s)) => {
                                nonce_count = 0;
                                Some(s)
                            }
                            Some(WorkerCommand::Close) => {
                                info!("GPU {}: Closing thread", gpu_work.id());
                                return Ok(());
                            }
                            None => None,
                        };
                    }
                }
                Ok(())
            })()
            .map_err(|e| {
                error!("GPU {}: Thread crashed: {}", gpu_work.id(), e);
                e
            })
        })
    }

    async fn log_hashrate(
        hashes_tried: Arc<AtomicU64>,
        cpu_hashes: Arc<AtomicU64>,
        hashes_by_worker: Arc<Mutex<HashMap<String, Arc<AtomicU64>>>>,
        thread_count: u16,
    ) {
        let mut ticker = tokio::time::interval(LOG_RATE);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let mut last_instant = ticker.tick().await;
        let mut last_total_hashes = 0;
        let mut last_cpu_hashes = 0;
        let mut last_gpu_hashes: HashMap<String, u64> = HashMap::new();

        loop {
            let now = ticker.tick().await;
            let duration = (now - last_instant).as_secs_f64();
            debug!("Logging hashrate, duration: {}s", duration);

            let total_hashes = hashes_tried.load(Ordering::Acquire);
            let total_diff = total_hashes.saturating_sub(last_total_hashes);
            let total_rate = (total_diff as f64) / duration;
            let (total_rate, total_suffix) = Self::hash_suffix(total_rate);

            let cpu_hashes_count = cpu_hashes.load(Ordering::Acquire);
            let cpu_diff = cpu_hashes_count.saturating_sub(last_cpu_hashes);
            let cpu_rate = (cpu_diff as f64) / duration;
            let (cpu_rate, cpu_suffix) = Self::hash_suffix(cpu_rate);

            let mut gpu_rates = Vec::new();
            let mut total_gpu_rate = 0.0;
            let mut gpu_count = 0;
            let workers = hashes_by_worker.lock().unwrap();
            for (device, counter) in workers.iter() {
                let hashes = counter.load(Ordering::Acquire);
                let last_hashes = *last_gpu_hashes.get(device).unwrap_or(&0);
                let diff = hashes.saturating_sub(last_hashes);
                let rate = (diff as f64) / duration;
                let (rate, suffix) = Self::hash_suffix(rate);
                total_gpu_rate += rate;
                gpu_rates.push((device.clone(), rate, suffix));
                gpu_count += 1;
                last_gpu_hashes.insert(device.clone(), hashes);
            }
            let (total_gpu_rate, _total_gpu_suffix) = Self::hash_suffix(total_gpu_rate);

            if total_diff == 0 {
                warn!(
                    "No hashes computed in the last {:.2}s. Check if node is synced or reduce workload.",
                    duration
                );
                if cpu_diff == 0 && thread_count > 0 {
                    warn!("CPU workers ({} threads) stalled. Check CPU load or configuration.", thread_count);
                }
                if gpu_count > 0 && total_gpu_rate == 0.0 {
                    warn!("GPU workers ({} devices) stalled. Check GPU drivers or workload settings.", gpu_count);
                }
            } else {
                info!(
                    "Total hashrate: {:.2} {} ({} CPU threads, {} GPUs)",
                    total_rate, total_suffix, thread_count, gpu_count
                );
                if cpu_diff > 0 {
                    info!("CPU hashrate ({} threads): {:.2} {}", thread_count, cpu_rate, cpu_suffix);
                } else if thread_count > 0 {
                    warn!("CPU workers ({} threads) not contributing. Check CPU configuration.", thread_count);
                }
                for (device, rate, suffix) in gpu_rates {
                    info!("GPU {} hashrate: {:.2} {}", device, rate, suffix);
                }
            }

            last_total_hashes = total_hashes;
            last_cpu_hashes = cpu_hashes_count;
            last_instant = now;
        }
    }

    fn hash_suffix(n: f64) -> (f64, &'static str) {
        match n {
            n if n < 1_000.0 => (n, "hash/s"),
            n if n < 1_000_000_000.0 => (n / 1_000_000.0, "Mhash/s"),
            n if n < 1_000_000_000_000.0 => (n / 1_000_000_000.0, "Ghash/s"),
            n if n < 1_000_000_000_000_000.0 => (n / 1_000_000_000_000.0, "Thash/s"),
            _ => (n, "hash/s"),
        }
    }
}