@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

nvcc plugins\cuda\vecno-cuda-native\src\vecno-cuda.cu -std=c++17 -O3 --restrict --ptx --gpu-architecture=compute_61 --gpu-code=sm_61 -o plugins\cuda\resources\vecno-cuda-sm61.ptx -Xptxas -O3 -Xcompiler /O2
if %ERRORLEVEL% NEQ 0 (
    echo Compilation for compute_61 failed!
    exit /b %ERRORLEVEL%
)

nvcc plugins\cuda\vecno-cuda-native\src\vecno-cuda.cu -std=c++17 -O3 --restrict --ptx --gpu-architecture=compute_75 --gpu-code=sm_75 -o plugins\cuda\resources\vecno-cuda-sm75.ptx -Xptxas -O3 -Xcompiler /O2
if %ERRORLEVEL% NEQ 0 (
    echo Compilation for compute_75 failed!
    exit /b %ERRORLEVEL%
)

nvcc plugins\cuda\vecno-cuda-native\src\vecno-cuda.cu -std=c++17 -O3 --restrict --ptx --gpu-architecture=compute_86 --gpu-code=sm_86 -o plugins\cuda\resources\vecno-cuda-sm86.ptx -Xptxas -O3 -Xcompiler /O2
if %ERRORLEVEL% NEQ 0 (
    echo Compilation for compute_86 failed!
    exit /b %ERRORLEVEL%
)

nvcc plugins\cuda\vecno-cuda-native\src\vecno-cuda.cu -std=c++17 -O3 --restrict --ptx --gpu-architecture=compute_89 --gpu-code=sm_89 -o plugins\cuda\resources\vecno-cuda-sm89.ptx -Xptxas -O3 -Xcompiler /O2
if %ERRORLEVEL% NEQ 0 (
    echo Compilation for compute_89 failed!
    exit /b %ERRORLEVEL%
)

nvcc plugins\cuda\vecno-cuda-native\src\vecno-cuda.cu -std=c++17 -O3 --restrict --ptx --gpu-architecture=compute_90 --gpu-code=sm_90 -o plugins\cuda\resources\vecno-cuda-sm90.ptx -Xptxas -O3 -Xcompiler /O2
if %ERRORLEVEL% NEQ 0 (
    echo Compilation for compute_90 failed!
    exit /b %ERRORLEVEL%
)

echo Compilation successful for all architectures!