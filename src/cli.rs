use clap::Parser;
use log::LevelFilter;
use std::error::Error;

#[derive(Parser, Debug)]
#[clap(name = "vecno-miner", version, about = "A high performance CPU/GPU miner", term_width = 0)]
pub struct Opt {
    #[clap(
        short = 'a',
        long = "mining-address",
        help = "The Vecno address for the miner reward (must start with 'vecno:')",
        validator = validate_mining_address
    )]
    pub mining_address: String,

    #[clap(
        short, long,
        help = "Enable debug logging level"
    )]
    pub debug: bool,

    #[clap(
        short = 's',
        long = "vecno-address",
        default_value = "127.0.0.1",
        help = "The IP of the vecno instance (used when not connecting to a stratum pool)"
    )]
    pub vecno_address: String,

    #[clap(
        short,
        long,
        help = "Vecnod port [default: Mainnet = 7110, Testnet = 7210]"
    )]
    pub port: Option<u16>,

    #[clap(long, help = "Use testnet instead of mainnet [default: false]")]
    pub testnet: bool,

    #[clap(
        short = 't',
        long = "threads",
        help = "Amount of CPU miner threads to launch [default: 0]"
    )]
    pub num_threads: Option<u16>,

    #[clap(
        long = "mine-when-not-synced",
        help = "Mine even when vecno says it is not synced",
        long_help = "Mine even when vecno says it is not synced, only useful when passing `--allow-submit-block-when-not-synced` to vecno [default: false]"
    )]
    pub mine_when_not_synced: bool,

    #[clap(long = "stratum-server", help = "Stratum pool server address (e.g., pool.example.com)")]
    pub stratum_server: Option<String>,

    #[clap(long = "stratum-port", help = "Stratum pool port [default: 6969]")]
    pub stratum_port: Option<u16>,

    #[clap(long = "stratum-worker", help = "Worker name for the stratum pool (e.g., worker1)")]
    pub stratum_worker: Option<String>,

    #[clap(long = "stratum-password", help = "Password for the stratum pool (optional)")]
    pub stratum_password: Option<String>,

    #[clap(long = "cuda-disable", help = "Disable CUDA workers")]
    pub cuda_disable: bool,

    #[clap(long = "cuda-device", help = "Which CUDA GPUs to use [default: all]")]
    pub cuda_device: Option<String>,

    #[clap(long = "cuda-lock-core-clocks", help = "Lock core clocks eg: ,1200, [default: 0]")]
    pub cuda_lock_core_clocks: Option<String>,

    #[clap(long = "cuda-lock-mem-clocks", help = "Lock mem clocks eg: ,810, [default: 0]")]
    pub cuda_lock_mem_clocks: Option<String>,

    #[clap(long = "cuda-power-limits", help = "Lock power limits eg: ,150, [default: 0]")]
    pub cuda_power_limits: Option<String>,

    #[clap(long = "cuda-no-blocking-sync", help = "Actively wait for result. Higher CPU usage, but less red blocks. Can have lower workload.")]
    pub cuda_no_blocking_sync: bool,

    #[clap(long = "cuda-workload-absolute", help = "The values given by workload are not ratio, but absolute number of nonces [default: false]")]
    pub cuda_workload_absolute: bool,

    #[clap(
        long = "cuda-nonce-gen",
        default_value = "lean",
        help = "The random method used to generate nonces. Options: (i) xoshiro (ii) lean [default: lean]",
        validator = validate_nonce_gen
    )]
    pub cuda_nonce_gen: String,

    #[clap(long = "opencl-amd-disable", help = "Disables AMD mining (does not override opencl-enable)")]
    pub opencl_amd_disable: bool,

    #[clap(long = "opencl-device", help = "Which OpenCL GPUs to use on a specific platform")]
    pub opencl_device: Option<String>,

    #[clap(long = "opencl-platform", help = "Which OpenCL platform to use (limited to one per executable)")]
    pub opencl_platform: Option<String>,

    #[clap(long = "opencl-no-amd-binary", help = "Disable fetching of precompiled AMD kernel (if exists)")]
    pub opencl_no_amd_binary: bool,

    #[clap(long = "opencl-workload-absolute", help = "The values given by workload are not ratio, but absolute number of nonces in OpenCL [default: false]")]
    pub opencl_workload_absolute: bool,

    #[clap(
        long = "opencl-nonce-gen",
        default_value = "lean",
        help = "The random method used to generate nonces. Options: (i) xoshiro (ii) lean [default: lean]",
        validator = validate_nonce_gen
    )]
    pub opencl_nonce_gen: String,

    #[clap(long = "cuda-workload", default_value = "128", help = "Ratio of nonces to GPU possible parallel run for CUDA [default: 128]")]
    pub cuda_workload: u32,

    #[clap(long = "opencl-workload", default_value = "128", help = "Ratio of nonces to GPU possible parallel run for OpenCL [default: 128]")]
    pub opencl_workload: u32,
}

// Validator function for mining_address
fn validate_mining_address(address: &str) -> Result<(), String> {
    if address.starts_with("vecno:") && address.len() > 6 {
        Ok(())
    } else {
        Err("Mining address must start with 'vecno:' and have a valid address".to_string())
    }
}

// Validator function for nonce_gen
fn validate_nonce_gen(nonce_gen: &str) -> Result<(), String> {
    if nonce_gen == "xoshiro" || nonce_gen == "lean" {
        Ok(())
    } else {
        Err("Nonce generation method must be 'xoshiro' or 'lean'".to_string())
    }
}

impl Opt {
    pub fn process(&mut self) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Validate mining_address
        if !self.mining_address.starts_with("vecno:") || self.mining_address.len() <= 6 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Mining address must start with 'vecno:' and have a valid address",
            )));
        }

        // Handle stratum configuration if provided
        if let Some(stratum_server) = &self.stratum_server {
            if self.stratum_worker.is_none() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Stratum worker name is required when using a stratum pool",
                )));
            }

            // Format vecno_address for stratum: stratum+tcp://<server>:<port>/<worker>
            let stratum_port = self.stratum_port.unwrap_or(6969);
            self.vecno_address = format!(
                "stratum+tcp://{}:{}/{}",
                stratum_server,
                stratum_port,
                self.stratum_worker.as_ref().unwrap()
            );
            log::info!("Stratum address: {}", self.vecno_address);

            // Stratum password is optional, log if provided
            if self.stratum_password.is_some() {
                log::info!("Stratum password provided");
            }
        } else {
            // Process vecno_address for non-stratum mode
            if self.vecno_address.is_empty() {
                self.vecno_address = "127.0.0.1".to_string();
            }
            if !self.vecno_address.contains("://") {
                let port_str = self.port().to_string();
                let (vecno, port) = match self.vecno_address.contains(':') {
                    true => self.vecno_address.split_once(':').expect("We checked for `:`"),
                    false => (self.vecno_address.as_str(), port_str.as_str()),
                };
                self.vecno_address = format!("grpc://{}:{}", vecno, port);
            }
            log::info!("Vecno address: {}", self.vecno_address);
        }

        // Set default number of threads if not specified
        if self.num_threads.is_none() {
            self.num_threads = Some(0);
        }

        Ok(())
    }

    pub fn port(&mut self) -> u16 {
        *self.port.get_or_insert(if self.testnet { 7210 } else { 7110 })
    }

    pub fn log_level(&self) -> LevelFilter {
        if self.debug {
            LevelFilter::Debug
        } else {
            LevelFilter::Info
        }
    }

    pub fn default() -> Self {
        Opt {
            debug: false,
            mining_address: String::new(),
            vecno_address: String::from("127.0.0.1"),
            port: None,
            testnet: false,
            num_threads: None,
            mine_when_not_synced: false,
            stratum_server: None,
            stratum_port: None,
            stratum_worker: None,
            stratum_password: None,
            cuda_disable: false,
            cuda_device: None,
            cuda_lock_core_clocks: None,
            cuda_lock_mem_clocks: None,
            cuda_power_limits: None,
            cuda_no_blocking_sync: false,
            cuda_workload_absolute: false,
            cuda_nonce_gen: String::from("lean"),
            opencl_amd_disable: false,
            opencl_device: None,
            opencl_platform: None,
            opencl_no_amd_binary: false,
            opencl_workload_absolute: false,
            opencl_nonce_gen: String::from("lean"),
            cuda_workload: 128,
            opencl_workload: 128,
        }
    }
}