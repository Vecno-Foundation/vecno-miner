#![cfg_attr(all(test, feature = "bench"), feature(test))]

use std::env::consts::DLL_EXTENSION;
use std::env::current_exe;
use std::error::Error as StdError;
use std::ffi::OsStr;
use clap::{App, FromArgMatches, IntoApp};
use vecno_miner::PluginManager;
use log::{error, info, debug, LevelFilter, warn};
use rand::{rng, Rng};
use std::fs::File;
use std::fs;
use std::sync::atomic::AtomicU16;
use std::sync::Arc;
use std::thread::sleep;
use std::time::Duration;
use simplelog::{WriteLogger, CombinedLogger, TermLogger, ConfigBuilder, TerminalMode, ColorChoice};
use simplelog::format_description;
use console::Term;
use crate::cli::Opt;
use crate::client::grpc::VecnodHandler;
use crate::client::stratum::StratumHandler;
use crate::client::Client;
use crate::miner::MinerManager;
use crate::target::Uint256;
use crate::interactivity::interactive_config;
use crate::pow::SHARE_TRACKER;

mod cli;
mod client;
mod vecnod_messages;
mod miner;
mod pow;
mod target;
mod watch;
mod interactivity;

const WHITELIST: [&str; 4] = ["libvecnocuda", "libvecnoopencl", "vecnocuda", "vecnoopencl"];

pub mod proto {
    #![allow(clippy::derive_partial_eq_without_eq)]
    tonic::include_proto!("protowire");
}

pub type Error = Box<dyn StdError + Send + Sync + 'static>;

pub type Hash = Uint256;

#[cfg(target_os = "windows")]
fn adjust_console() -> Result<(), Error> {
    use win32console::console::{WinConsole, ConsoleMode};

    let console = WinConsole::output();
    let mode = match console.get_mode() {
        Ok(m) => m,
        Err(e) => {
            debug!("Failed to get console mode: {}. Proceeding with defaults.", e);
            return Ok(());
        }
    };

    debug!("Current console mode: {:?}", mode);

    let new_mode = (mode & !ConsoleMode::ENABLE_PROCESSED_INPUT)
        | ConsoleMode::ENABLE_EXTENDED_FLAGS
        | ConsoleMode::ENABLE_QUICK_EDIT_MODE;

    if new_mode == mode {
        debug!("Console mode already optimal.");
        return Ok(());
    }

    if let Err(e) = console.set_mode(new_mode) {
        if e.raw_os_error() == Some(87) {
            warn!("Console does not support requested mode (error 87). Text selection may still work, but miner might pause on click.");
            debug!("Attempted mode: {:?}, Current: {:?}", new_mode, mode);
        } else {
            warn!("Failed to set console mode: {}. Miner might pause on interaction.", e);
        }
    } else {
        debug!("Console mode updated successfully.");
    }

    Ok(())
}

fn filter_plugins(dirname: &str) -> Vec<String> {
    match fs::read_dir(dirname) {
        Ok(readdir) => readdir
            .map(|entry| entry.unwrap().path())
            .filter(|fname| {
                fname.is_file()
                    && fname.extension().is_some()
                    && fname.extension().and_then(OsStr::to_str).unwrap_or_default().starts_with(DLL_EXTENSION)
            })
            .filter(|fname| WHITELIST.iter().any(|lib| *lib == fname.file_stem().and_then(OsStr::to_str).unwrap()))
            .map(|path| path.to_str().unwrap().to_string())
            .collect::<Vec<String>>(),
        _ => Vec::<String>::new(),
    }
}

async fn get_client(
    vecno_address: String,
    mining_address: String,
    mine_when_not_synced: bool,
    block_template_ctr: Arc<AtomicU16>,
    stratum_password: Option<String>,
) -> Result<Box<dyn Client + 'static>, Error> {
    if vecno_address.starts_with("stratum+tcp://") {
        let (_schema, address) = vecno_address.split_once("://").unwrap();
        Ok(StratumHandler::connect(
            address.to_string(),
            mining_address.clone(),
            mine_when_not_synced,
            Some(block_template_ctr.clone()),
            stratum_password.clone(),
        )
        .await?)
    } else if vecno_address.starts_with("grpc://") {
        Ok(VecnodHandler::connect(
            vecno_address.clone(),
            mining_address.clone(),
            mine_when_not_synced,
            Some(block_template_ctr.clone()),
        )
        .await?)
    } else {
        Err("Did not recognize pool/grpc address schema".into())
    }
}

async fn client_main(
    opt: &Opt,
    block_template_ctr: Arc<AtomicU16>,
    plugin_manager: &PluginManager,
) -> Result<(), Error> {
    let mut client = get_client(
        opt.vecno_address.clone(),
        opt.mining_address.clone(),
        opt.mine_when_not_synced,
        block_template_ctr.clone(),
        opt.stratum_password.clone(),
    )
    .await?;

    client.register().await?;
    let mut miner_manager = MinerManager::new(client.get_block_channel(), opt.num_threads, plugin_manager);
    client.listen(&mut miner_manager).await?;
    drop(miner_manager);
    Ok(())
}

fn setup_logger(log_level: LevelFilter) -> Result<(), Error> {
    let log_file = File::create("debug.log")?;
    let mut fallback_builder = ConfigBuilder::new();
    let config = ConfigBuilder::new()
        .set_time_format_custom(format_description!("[hour]:[minute]:[second]"))
        .set_time_offset_to_local()
        .unwrap_or(&mut fallback_builder)
        .set_location_level(LevelFilter::Off)
        .build();

    CombinedLogger::init(vec![
        TermLogger::new(
            log_level,
            config.clone(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        ),
        WriteLogger::new(
            log_level,
            config,
            log_file,
        ),
    ])?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    #[cfg(target_os = "windows")]
    adjust_console().unwrap_or_else(|e| {
        warn!("Console setup failed ({}). Selection may work, but clicking might pause miner.", e);
    });

    let term = Term::stdout();

    let mut path = current_exe().unwrap_or_default();
    path.pop();
    let plugins = filter_plugins(path.to_str().unwrap_or("."));
    debug!("Detected plugins: {:?}", plugins);
    let (app, mut plugin_manager): (App, PluginManager) = vecno_miner::load_plugins(Opt::into_app(), &plugins)?;

    let matches = app.clone().try_get_matches();
    let mut opt = match matches {
        Ok(matches) => Opt::from_arg_matches(&matches)?,
        Err(_) => interactive_config(&term, &plugins)?,
    };

    let needs_interactive = opt.mining_address.is_empty() || (opt.stratum_server.is_none() && opt.vecno_address.is_empty());
    if needs_interactive {
        opt = interactive_config(&term, &plugins)?;
    }
    
    opt.process()?;
    let port = opt.port();
    debug!("Resolved port: {}", port);

    let mut args = vec!["vecno-miner".to_string()];
    args.push("--mining-address".to_string());
    args.push(opt.mining_address.clone());
    if opt.cuda_disable {
        args.push("--cuda-disable".to_string());
    } else {
        args.push(format!("--cuda-workload={}", opt.cuda_workload));
        if let Some(cuda_device) = &opt.cuda_device {
            args.push(format!("--cuda-device={}", cuda_device));
        }
        args.push(format!("--cuda-lock-core-clocks={}", opt.cuda_lock_core_clocks.as_ref().unwrap_or(&"0".to_string())));
        args.push(format!("--cuda-lock-mem-clocks={}", opt.cuda_lock_mem_clocks.as_ref().unwrap_or(&"0".to_string())));
        args.push(format!("--cuda-power-limits={}", opt.cuda_power_limits.as_ref().unwrap_or(&"0".to_string())));
        if opt.cuda_no_blocking_sync {
            args.push("--cuda-no-blocking-sync".to_string());
        }
        if opt.cuda_workload_absolute {
            args.push("--cuda-workload-absolute".to_string());
        }
        args.push(format!("--cuda-nonce-gen={}", opt.cuda_nonce_gen));
    }
    if !opt.opencl_amd_disable {
        args.push(format!("--opencl-workload={}", opt.opencl_workload));
        if let Some(opencl_device) = &opt.opencl_device {
            args.push(format!("--opencl-device={}", opencl_device));
        }
        if let Some(opencl_platform) = &opt.opencl_platform {
            args.push(format!("--opencl-platform={}", opencl_platform));
        }
        if opt.opencl_no_amd_binary {
            args.push("--opencl-no-amd-binary".to_string());
        }
        if opt.opencl_workload_absolute {
            args.push("--opencl-workload-absolute".to_string());
        }
        args.push(format!("--opencl-nonce-gen={}", opt.opencl_nonce_gen));
    }
    if opt.opencl_amd_disable {
        args.push("--opencl-amd-disable".to_string());
    }

    debug!("Generated clap arguments for PluginManager: {:?}", args);
    let matches = Opt::into_app().get_matches_from(args.clone());

    let worker_count = plugin_manager.process_options(&matches)?;
    info!("GPU worker count after processing options: {}", worker_count);

    setup_logger(opt.log_level())?;

    info!("=================================================================================");
    info!("                 Vecno-Miner GPU version {}", env!("CARGO_PKG_VERSION"));
    info!(" Mining for: {}", opt.mining_address);
    info!(" Pool/Node: {}", opt.vecno_address);
    info!(" GPU Mining: {}", if !opt.cuda_disable || !opt.opencl_amd_disable { "Enabled" } else { "Disabled" });
    info!(" CUDA Enabled: {}", !opt.cuda_disable);
    info!(" OpenCL Enabled: {}", !opt.opencl_amd_disable);
    info!("=================================================================================");
    info!("Plugins found {} workers", worker_count);
    if worker_count == 0 && opt.num_threads.unwrap_or_default() == 0 {
        error!("No workers specified. Please specify at least one worker (CPU or GPU).");
        error!("CUDA disable: {}, OpenCL AMD disable: {}", opt.cuda_disable, opt.opencl_amd_disable);
        error!("Plugins detected: {:?}", plugins);
        error!("Arguments passed to PluginManager: {:?}", args);
        error!("Recommendation: Enable debug logging (--debug) to inspect PluginManager behavior.");
        return Err("No workers specified".into());
    }

    SHARE_TRACKER.start_reporter();

    let mut rng = rng();
    let block_template_ctr = Arc::new(AtomicU16::new(rng.random_range(0..10_000)));
    loop {
        match client_main(&opt, block_template_ctr.clone(), &plugin_manager).await {
            Ok(_) => info!("Client closed gracefully"),
            Err(e) => error!("Client closed with error: {:?}", e),
        }
        info!("Client connection closed, attempting to reconnect");
        sleep(Duration::from_millis(100));
    }
}