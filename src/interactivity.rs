use console::{style, Term};
use dialoguer::{theme::ColorfulTheme, Input, Confirm, Password, Select};
use std::error::Error as StdError;
use crate::cli::Opt;

pub fn interactive_config(term: &Term, plugins: &[String]) -> Result<Opt, Box<dyn StdError + Send + Sync + 'static>> {
    let theme = ColorfulTheme::default();
    let mut opt = Opt::default();

    loop {
        // Welcome screen
        term.clear_screen()?;
        term.write_line(&format!(
            "{}\n",
            style("=============================================================").bold().cyan()
        ))?;
        term.write_line(&format!(
            "{}\n",
            style(format!(" Welcome to Vecno-Miner v{}", env!("CARGO_PKG_VERSION"))).bold().green()
        ))?;
        term.write_line(&format!(
            "{}\n",
            style("=============================================================").bold().cyan()
        ))?;
        term.write_line("This is an interactive setup for Vecno-Miner. You'll be guided through configuring the miner.\n")?;
        term.write_line(&format!(
            "Press {} to accept defaults, or provide your own values.\n",
            style("Enter").bold()
        ))?;
        term.write_line(&format!("{}\n", style("Let's get started!").italic()))?;
        term.write_line("\n")?;

        // Section: Mining Address
        term.write_line(&format!(
            "{}\n",
            style("[1/4] Mining Address Configuration").bold().yellow()
        ))?;
        term.write_line("Your mining address is where your rewards will be sent. It must start with 'vecno:'.\n")?;
        opt.mining_address = Input::with_theme(&theme)
            .with_prompt("Enter your Vecno mining address")
            .validate_with(|input: &String| -> Result<(), &str> {
                if input.starts_with("vecno:") && input.len() > 6 {
                    Ok(())
                } else {
                    Err("Mining address must start with 'vecno:' and have a valid address")
                }
            })
            .interact_text()?;

        // Section: Connection Settings
        term.write_line(&format!(
            "{}\n",
            style("[2/4] Connection Settings").bold().yellow()
        ))?;
        term.write_line("Choose whether to connect to a stratum pool or a Vecno node.\n")?;
        let connection_options = vec!["Stratum Pool", "Vecno Node"];
        let connection_choice = Select::with_theme(&theme)
            .with_prompt("Select connection type")
            .items(&connection_options)
            .default(0)
            .interact()?;

        if connection_choice == 0 {
            // Stratum Pool Configuration
            term.write_line("Configuring Stratum pool connection...\n")?;
            opt.stratum_server = Some(Input::with_theme(&theme)
                .with_prompt("Enter stratum pool server address (e.g., vecnopool.com or vecnopool.de)")
                .default("vecnopool.de".to_string())
                .validate_with(|input: &String| -> Result<(), &str> {
                    if !input.is_empty() {
                        Ok(())
                    } else {
                        Err("Stratum server address cannot be empty")
                    }
                })
                .interact_text()?);

            let default_stratum_port = 6969;
            let stratum_port_input: String = Input::with_theme(&theme)
                .with_prompt(format!("Enter stratum pool port (default: {})", default_stratum_port))
                .default(default_stratum_port.to_string())
                .allow_empty(true)
                .interact_text()?;
            opt.stratum_port = if stratum_port_input.is_empty() {
                Some(default_stratum_port)
            } else {
                Some(stratum_port_input.parse::<u16>().map_err(|_| "Invalid stratum port")?)
            };

            opt.stratum_worker = Some(Input::with_theme(&theme)
                .with_prompt("Enter stratum worker name (e.g., worker1)")
                .validate_with(|input: &String| -> Result<(), &str> {
                    if !input.is_empty() {
                        Ok(())
                    } else {
                        Err("Stratum worker name cannot be empty")
                    }
                })
                .interact_text()?);

            if Confirm::with_theme(&theme)
                .with_prompt("Do you want to set a stratum password?")
                .default(false)
                .interact()? {
                opt.stratum_password = Some(Password::with_theme(&theme)
                    .with_prompt("Enter stratum password")
                    .interact()?);
            }
            opt.vecno_address = format!("stratum+tcp://{}:{}", opt.stratum_server.as_ref().unwrap(), opt.stratum_port.unwrap());
        } else {
            // Vecno Node Configuration
            term.write_line("Configuring Vecno node connection...\n")?;
            opt.vecno_address = Input::with_theme(&theme)
                .with_prompt("Enter Vecno node address (e.g., 127.0.0.1)")
                .default("127.0.0.1".to_string())
                .interact_text()?;

            opt.testnet = Confirm::with_theme(&theme)
                .with_prompt("Use testnet? (default: no)")
                .default(false)
                .interact()?;

            let default_port = if opt.testnet { 7210 } else { 7110 };
            let port_input: String = Input::with_theme(&theme)
                .with_prompt(format!("Enter Vecno node port (default: {})", default_port))
                .default(default_port.to_string())
                .allow_empty(true)
                .interact_text()?;
            opt.port = if port_input.is_empty() {
                Some(default_port)
            } else {
                Some(port_input.parse::<u16>().map_err(|_| "Invalid port")?)
            };
            opt.vecno_address = format!("grpc://{}:{}", opt.vecno_address, opt.port.unwrap());
        }

        // Section: Mining Hardware
        term.write_line(&format!(
            "{}\n",
            style("[3/4] Mining Hardware Configuration").bold().yellow()
        ))?;
        if !plugins.is_empty() {
            let enable_gpu = Confirm::with_theme(&theme)
                .with_prompt(format!("Enable GPU mining? (Detected plugins: {})", plugins.join(", ")))
                .default(true)
                .interact()?;
            if enable_gpu {
                let has_cuda = plugins.iter().any(|p| p.contains("cuda"));
                let enable_cuda = if has_cuda {
                    Confirm::with_theme(&theme)
                        .with_prompt("Enable CUDA for NVIDIA GPUs?")
                        .default(true)
                        .interact()?
                } else {
                    false
                };

                if enable_cuda {
                    opt.cuda_disable = false;
                    let use_cuda_custom = Confirm::with_theme(&theme)
                        .with_prompt("Customize CUDA settings?")
                        .default(false)
                        .interact()?;
                    if use_cuda_custom {
                        let cuda_device_input: String = Input::with_theme(&theme)
                            .with_prompt("Enter CUDA device IDs (e.g., 0,1,2 or empty for all)")
                            .allow_empty(true)
                            .interact_text()?;
                        if !cuda_device_input.is_empty() {
                            opt.cuda_device = Some(cuda_device_input);
                        }
                        let cuda_workload_input: String = Input::with_theme(&theme)
                            .with_prompt("Enter CUDA workload (nonce ratio, default: 128)")
                            .default("128".to_string())
                            .interact_text()?;
                        opt.cuda_workload = cuda_workload_input.parse::<u32>().map_err(|_| "Invalid CUDA workload")?;
                        let cuda_lock_core_clocks_input: String = Input::with_theme(&theme)
                            .with_prompt("Enter CUDA core clock locks (e.g., ,1200, or empty for default: 0)")
                            .allow_empty(true)
                            .interact_text()?;
                        opt.cuda_lock_core_clocks = if cuda_lock_core_clocks_input.is_empty() {
                            Some("0".to_string())
                        } else {
                            Some(cuda_lock_core_clocks_input)
                        };
                        let cuda_lock_mem_clocks_input: String = Input::with_theme(&theme)
                            .with_prompt("Enter CUDA memory clock locks (e.g., ,810, or empty for default: 0)")
                            .allow_empty(true)
                            .interact_text()?;
                        opt.cuda_lock_mem_clocks = if cuda_lock_mem_clocks_input.is_empty() {
                            Some("0".to_string())
                        } else {
                            Some(cuda_lock_mem_clocks_input)
                        };
                        let cuda_power_limits_input: String = Input::with_theme(&theme)
                            .with_prompt("Enter CUDA power limits (e.g., ,150, or empty for default: 0)")
                            .allow_empty(true)
                            .interact_text()?;
                        opt.cuda_power_limits = if cuda_power_limits_input.is_empty() {
                            Some("0".to_string())
                        } else {
                            Some(cuda_power_limits_input)
                        };
                        opt.cuda_no_blocking_sync = Confirm::with_theme(&theme)
                            .with_prompt("Enable CUDA no-blocking sync? (Higher CPU usage, fewer red blocks)")
                            .default(false)
                            .interact()?;
                        opt.cuda_workload_absolute = Confirm::with_theme(&theme)
                            .with_prompt("Use absolute CUDA workload values instead of ratio?")
                            .default(false)
                            .interact()?;
                        let nonce_gen_options = vec!["lean", "xoshiro"];
                        let cuda_nonce_gen_index = Select::with_theme(&theme)
                            .with_prompt("Select CUDA nonce generation method")
                            .items(&nonce_gen_options)
                            .default(0)
                            .interact()?;
                        opt.cuda_nonce_gen = nonce_gen_options[cuda_nonce_gen_index].to_string();
                    }
                } else {
                    opt.cuda_disable = true;
                }

                let has_opencl = plugins.iter().any(|p| p.contains("opencl"));
                let enable_opencl = if has_opencl {
                    Confirm::with_theme(&theme)
                        .with_prompt("Enable OpenCL for AMD GPUs or other devices?")
                        .default(true)
                        .interact()?
                } else {
                    false
                };

                if enable_opencl {
                    let use_opencl_custom = Confirm::with_theme(&theme)
                        .with_prompt("Customize OpenCL settings?")
                        .default(false)
                        .interact()?;
                    if use_opencl_custom {
                        let disable_amd_opencl = Confirm::with_theme(&theme)
                            .with_prompt("Disable AMD-specific OpenCL mining?")
                            .default(false)
                            .interact()?;
                        opt.opencl_amd_disable = disable_amd_opencl;
                        let opencl_platform_input: String = Input::with_theme(&theme)
                            .with_prompt("Enter OpenCL platform ID (e.g., 0, or empty for default)")
                            .allow_empty(true)
                            .interact_text()?;
                        if !opencl_platform_input.is_empty() {
                            opt.opencl_platform = Some(opencl_platform_input);
                        }
                        let opencl_device_input: String = Input::with_theme(&theme)
                            .with_prompt("Enter OpenCL device IDs (e.g., 0,1,2 or empty for all)")
                            .allow_empty(true)
                            .interact_text()?;
                        if !opencl_device_input.is_empty() {
                            opt.opencl_device = Some(opencl_device_input);
                        }
                        let opencl_workload_input: String = Input::with_theme(&theme)
                            .with_prompt("Enter OpenCL workload (nonce ratio, default: 128)")
                            .default("128".to_string())
                            .interact_text()?;
                        opt.opencl_workload = opencl_workload_input.parse::<u32>().map_err(|_| "Invalid OpenCL workload")?;
                        opt.opencl_no_amd_binary = Confirm::with_theme(&theme)
                            .with_prompt("Disable fetching of precompiled AMD kernel for OpenCL?")
                            .default(false)
                            .interact()?;
                        opt.opencl_workload_absolute = Confirm::with_theme(&theme)
                            .with_prompt("Use absolute OpenCL workload values instead of ratio?")
                            .default(false)
                            .interact()?;
                        let nonce_gen_options = vec!["lean", "xoshiro"];
                        let opencl_nonce_gen_index = Select::with_theme(&theme)
                            .with_prompt("Select OpenCL nonce generation method")
                            .items(&nonce_gen_options)
                            .default(0)
                            .interact()?;
                        opt.opencl_nonce_gen = nonce_gen_options[opencl_nonce_gen_index].to_string();
                    } else {
                        opt.opencl_amd_disable = false;
                    }
                } else {
                    opt.opencl_amd_disable = true;
                }

                if !enable_cuda && !enable_opencl {
                    term.write_line(&format!(
                        "{}\n",
                        style("Warning: Neither CUDA nor OpenCL enabled. GPU mining disabled.").yellow()
                    ))?;
                    opt.cuda_disable = true;
                    opt.opencl_amd_disable = true;
                }
            } else {
                opt.cuda_disable = true;
                opt.opencl_amd_disable = true;
                term.write_line(&format!(
                    "{}\n",
                    style("GPU mining disabled.").yellow()
                ))?;
            }
        } else {
            opt.cuda_disable = true;
            opt.opencl_amd_disable = true;
            term.write_line(&format!(
                "{}\n",
                style("No GPU plugins found (libvecnocuda or libvecnoopencl). GPU mining disabled.").yellow()
            ))?;
        }

        // CPU Threads Configuration
        let max_threads = num_cpus::get() as u16;
        let default_threads = if !opt.cuda_disable && !opt.opencl_amd_disable {
            max_threads.saturating_sub(4)
        } else if !opt.cuda_disable || !opt.opencl_amd_disable {
            max_threads.saturating_sub(2)
        } else {
            max_threads
        };
        let threads_input: String = Input::with_theme(&theme)
            .with_prompt(format!("Enter number of CPU threads (default: {}, max: {})", default_threads, max_threads))
            .default(default_threads.to_string())
            .allow_empty(true)
            .interact_text()?;
        opt.num_threads = if threads_input.is_empty() {
            Some(default_threads)
        } else {
            let threads = threads_input.parse::<u16>().map_err(|_| "Invalid number of threads")?;
            if threads > max_threads {
                term.write_line(&format!(
                    "{}\n",
                    style(format!(
                        "Warning: Specified threads ({}) exceed available ({}). Using {}.",
                        threads, max_threads, max_threads
                    )).yellow()
                ))?;
                Some(max_threads)
            } else {
                Some(threads)
            }
        };

        // Section: Advanced Settings
        term.write_line(&format!(
            "{}\n",
            style("[4/4] Advanced Settings").bold().yellow()
        ))?;
        opt.mine_when_not_synced = Confirm::with_theme(&theme)
            .with_prompt("Mine even when not synced? (default: no)")
            .default(false)
            .interact()?;

        opt.debug = Confirm::with_theme(&theme)
            .with_prompt("Enable debug logging for detailed output? (default: no)")
            .default(false)
            .interact()?;

        // Summary and Confirmation
        term.write_line(&format!(
            "{}\n",
            style("Configuration Summary").bold().cyan()
        ))?;
        term.write_line(&format!("Mining Address: {}\n", style(&opt.mining_address).green()))?;
        term.write_line(&format!("Connection: {}\n", style(&opt.vecno_address).green()))?;
        term.write_line(&format!(
            "GPU Mining: {}\n",
            style(if !opt.cuda_disable || !opt.opencl_amd_disable { "Enabled" } else { "Disabled" }).green()
        ))?;
        term.write_line(&format!("CUDA Enabled: {}\n", style(!opt.cuda_disable).green()))?;
        term.write_line(&format!("OpenCL Enabled: {}\n", style(!opt.opencl_amd_disable).green()))?;
        term.write_line(&format!("CPU Threads: {}\n", style(opt.num_threads.unwrap_or(0)).green()))?;
        term.write_line(&format!("Mine When Not Synced: {}\n", style(opt.mine_when_not_synced).green()))?;
        term.write_line(&format!("Debug Logging: {}\n", style(opt.debug).green()))?;

        if Confirm::with_theme(&theme)
            .with_prompt("Is this configuration correct? (Select 'No' to restart configuration)")
            .default(true)
            .interact()?
        {
            break; // Exit the configuration loop
        }
        term.write_line("Restarting configuration...\n")?;
    }

    Ok(opt)
}