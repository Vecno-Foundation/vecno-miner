# Cuda Support For Vecno-Miner

## Building

The plugin is a shared library file that resides in the same library as the miner. 
You can build the library by running
```sh
cargo build -p vecnocuda
```

This version includes a precompiled PTX, which would work with most modern GPUs. To compile the PTX youself,
you have to clone the project:

```sh
git clone https://github.com/vecno-foundation/vecno-miner.git
cd vecno-miner
# Compute version 8.6
/usr/local/cuda-11.5/bin/nvcc plugins/cuda/vecno-cuda-native/src/vecno-cuda.cu -std=c++11 -O3 --restrict --ptx --gpu-architecture=compute_86 --gpu-code=sm_86 -o plugins/cuda/resources/vecno-cuda-sm86.ptx -Xptxas -O3 -Xcompiler -O3
# Compute version 7.5
/usr/local/cuda-11.5/bin/nvcc plugins/cuda/vecno-cuda-native/src/vecno-cuda.cu -std=c++11 -O3 --restrict --ptx --gpu-architecture=compute_75 --gpu-code=sm_75 -o plugins/cuda/resources/vecno-cuda-sm75.ptx -Xptxas -O3 -Xcompiler -O3
# Compute version 6.1
/usr/local/cuda-11.2/bin/nvcc plugins/cuda/vecno-cuda-native/src/vecno-cuda.cu -std=c++11 -O3 --restrict --ptx --gpu-architecture=compute_61 --gpu-code=sm_61 -o plugins/cuda/resources/vecno-cuda-sm61.ptx -Xptxas -O3 -Xcompiler -O3
# Compute version 3.0
/usr/local/cuda-9.2/bin/nvcc plugins/cuda/vecno-cuda-native/src/vecno-cuda.cu -ccbin=gcc-7 -std=c++11 -O3 --restrict --ptx --gpu-architecture=compute_30 --gpu-code=sm_30 -o plugins/cuda/resources/vecno-cuda-sm30.ptx
# Compute version 2.0
/usr/local/cuda-8.0/bin/nvcc plugins/cuda/vecno-cuda-native/src/vecno-cuda.cu -ccbin=gcc-5 -std=c++11 -O3 --restrict --ptx --gpu-architecture=compute_20 --gpu-code=sm_20 -o plugins/cuda/resources/vecno-cuda-sm20.ptx
 
cargo build --release
```
