# Vecno-miner

This miner supports Nvidia and AMD GPU's, along with mining with your CPU.

## Installation

### From Git Sources

If you are looking to build from the repository (for debug / extension), note that the plugins are additional
packages in the workspace. To compile a specific package, you run the following command or any subset of it

```sh
git clone git@github.com:vecno-foundation/vecno-miner.git
cd vecno-miner
cargo build --release -p vecno-miner -p vecnocuda -p vecnoopencl
```

And, the miner (and plugins) will be in `targets/release`. You can replace the last line with

```sh
cargo build --release --all
```

### From Binaries

The [release page](https://github.com/Vecno-Foundation/vecno-miner/releases) includes precompiled binaries for Linux, and Windows (for the GPU version).

### Removing Plugins

To remove a plugin, you simply remove the corresponding `dll`/`so` for the directory of the miner.

* `libvecnocuda.so`, `libvecnocuda.dll`: Cuda support for Vecno-Miner
* `libvecnoopencl.so`, `libvecnoopencl.dll`: OpenCL support for Vecno-Miner

# Usage

To start mining, you need to run [vecno](https://github.com/Vecno-Foundation/vecnod) or mine using a startum pool. You need have an address to send the rewards to.

Help:

```
vecno-miner 
A Vecno high performance CPU/GPU miner

USAGE:
    vecno-miner [OPTIONS] --mining-address <MINING_ADDRESS>

OPTIONS:
    -a, --mining-address <MINING_ADDRESS>                  The Vecno address for the miner reward [vecno:xxxx]
        --cuda-device <CUDA_DEVICE>                        Which CUDA GPUs to use [default: all]
        --cuda-disable                                     Disable cuda workers
        --cuda-lock-core-clocks <CUDA_LOCK_CORE_CLOCKS>    Lock core clocks eg: ,1200, [default: 0]
        --cuda-lock-mem-clocks <CUDA_LOCK_MEM_CLOCKS>      Lock mem clocks eg: ,810, [default: 0]
        --cuda-no-blocking-sync                            Actively wait for result. Higher CPU usage, but less red blocks. Can have lower workload.
        --cuda-power-limits <CUDA_POWER_LIMITS>            Lock power limits eg: ,150, [default: 0]
        --cuda-workload <CUDA_WORKLOAD>                    Ratio of nonces to GPU possible parrallel run [default: 128]
        --cuda-workload-absolute                           The values given by workload are not ratio, but absolute number of nonces [default: false]
	--cuda-nonce-gen <NONCE_GEN>                       The random method used to generate nonces. Options: (i) xoshiro (ii) lean [default: lean]
    -d, --debug                                            Enable debug logging level
    -h, --help                                             Print help information
        --mine-when-not-synced                             Mine even when vecno says it is not synced
        --opencl-nonce-gen <NONCE_GEN>                     The random method used to generate nonces. Options: (i) xoshiro (ii) lean [default: lean]
        --opencl-amd-disable                               Disables AMD mining (does not override opencl-enable)
        --opencl-device <OPENCL_DEVICE>                    Which OpenCL GPUs to use on a specific platform
        --opencl-no-amd-binary                             Disable fetching of precompiled AMD kernel (if exists)
        --opencl-platform <OPENCL_PLATFORM>                Which OpenCL platform to use (limited to one per executable)
        --opencl-workload <OPENCL_WORKLOAD>                Ratio of nonces to GPU possible parrallel run in OpenCL [default: 128]
        --opencl-workload-absolute                         The values given by workload are not ratio, but absolute number of nonces in OpenCL [default: false]
    -p, --port <PORT>                                      Vecnod port [default: Mainnet = 7110]
    -s, --vecno-address <VECNO_ADDRESS>                    The IP of the vecno instance [default: 127.0.0.1]
    -t, --threads <NUM_THREADS>                            Amount of CPU miner threads to launch [default: 0]

STRATUM POOL:
	--stratum-server <STRATUM_ADDRESS>		   The Stratum address for mining
	--stratum-port <STRATUM_PORT>                      Stratum port
	--stratum-worker <WORKER_NAME>                     Worker name 
	--stratum-password <WORKER_PASSWORD>               Worker password [optional]

```



# Solo mining

To start SOLO mining , you just need to run the following:

```
./vecno-miner --vecno-address <VECNO_ADDRESS> --mining-address <MINING_ADDRESS>
```

This will run the miner on all the available GPU devcies.

Optional: use arg `--threads ` to activate CPU mining. This uses all available CPU threads.

Solo mining requires you ruinning your own node.


# Mining Pool

```
./vecno-miner --mining-address <MINING_ADDRESS> --stratum-server <STRATUM_ADDRESS> --stratum-port <STRATUM_PORT> --stratum-worker <WORKER_NAME> --stratum-password <WORKER_PASSWORD>
```

Running this will activate all available GPU devcies.

Optional: use arg `--threads ` to activate CPU mining. This uses all available CPU threads.

## Support

This mining software is experimental, testnet mining is disabled. Use [vecno-cpu-miner](https://github.com/Vecno-Foundation/vecno-cpu-miner) if you want to need to mine on testnet.
