@echo off
setlocal EnableDelayedExpansion

call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if errorlevel 1 call "C:\Program Files\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if errorlevel 1 (
    echo ERROR: Visual Studio vcvars64.bat not found! Install VS 2022 or 2019 Community.
    pause
    exit /b 1
)

echo.
echo ====================================================
echo     Vecno CUDA Miner - Multi-Architecture Build
echo ====================================================
echo.

set UCRT_INCLUDE="C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt"

set COMMON_FLAGS= -std=c++17 -O3 --restrict --use_fast_math --extra-device-vectorization -lineinfo ^
                  -Xptxas -O3,-dlcm=cg,-v,--allow-expensive-optimizations ^
                  -Xcompiler /O2,/MD,/wd4819 ^
                  -Wno-deprecated-gpu-targets ^
                  --allow-unsupported-compiler ^
                  -I%UCRT_INCLUDE%

echo Compiling for GTX 10xx (Pascal)       - sm_61
nvcc plugins\cuda\vecno-cuda-native\src\vecno-cuda.cu %COMMON_FLAGS% --ptx --gpu-architecture=compute_61 --gpu-code=sm_61 -o plugins\cuda\resources\vecno-cuda-sm61.ptx
if !ERRORLEVEL! NEQ 0 echo ERROR: sm_61 failed && pause && exit /b !ERRORLEVEL!

echo Compiling for GTX 16xx / RTX 20xx (Turing) - sm_75
nvcc plugins\cuda\vecno-cuda-native\src\vecno-cuda.cu %COMMON_FLAGS% --ptx --gpu-architecture=compute_75 --gpu-code=sm_75 -o plugins\cuda\resources\vecno-cuda-sm75.ptx
if !ERRORLEVEL! NEQ 0 echo ERROR: sm_75 failed && pause && exit /b !ERRORLEVEL!

echo Compiling for RTX 20xx Super / Titan (Turing+) - sm_75 (same as above)

echo Compiling for RTX 30xx (Ampere)       - sm_80 + sm_86
nvcc plugins\cuda\vecno-cuda-native\src\vecno-cuda.cu %COMMON_FLAGS% --ptx --gpu-architecture=compute_80 --gpu-code=sm_80 -o plugins\cuda\resources\vecno-cuda-sm80.ptx
if !ERRORLEVEL! NEQ 0 echo ERROR: sm_80 failed && pause && exit /b !ERRORLEVEL!

nvcc plugins\cuda\vecno-cuda-native\src\vecno-cuda.cu %COMMON_FLAGS% --ptx --gpu-architecture=compute_86 --gpu-code=sm_86 -o plugins\cuda\resources\vecno-cuda-sm86.ptx
if !ERRORLEVEL! NEQ 0 echo ERROR: sm_86 failed && pause && exit /b !ERRORLEVEL!

echo Compiling for RTX 40xx (Ada Lovelace) - sm_89
nvcc plugins\cuda\vecno-cuda-native\src\vecno-cuda.cu %COMMON_FLAGS% --ptx --gpu-architecture=compute_89 --gpu-code=sm_89 -o plugins\cuda\resources\vecno-cuda-sm89.ptx
if !ERRORLEVEL! NEQ 0 echo ERROR: sm_89 failed && pause && exit /b !ERRORLEVEL!

echo Compiling for RTX 50xx (Blackwell)    - sm_90  [Requires CUDA 12.4+]
nvcc plugins\cuda\vecno-cuda-native\src\vecno-cuda.cu %COMMON_FLAGS% --ptx --gpu-architecture=compute_90 --gpu-code=sm_90 -o plugins\cuda\resources\vecno-cuda-sm90.ptx
set SM90_ERROR=!ERRORLEVEL!
if !SM90_ERROR! NEQ 0 (
    echo WARNING: sm_90 (Blackwell) failed — this is OK if you don't have CUDA 12.4+ or newer installed.
    echo          RTX 50-series support will be disabled.
) else (
    echo SUCCESS: Blackwell (sm_90) compiled!
)

echo.
echo ====================================================
echo  ALL SUPPORTED ARCHITECTURES COMPILED SUCCESSFULLY!
echo.
echo  Supported GPUs:
echo    - GTX 10xx series         (sm_61)
echo    - GTX 16xx / RTX 20xx     (sm_75)
echo    - RTX 30xx series         (sm_80/sm_86)
echo    - RTX 40xx series         (sm_89)
echo    - RTX 50xx series         (sm_90)  [if CUDA 12.4+ installed]
echo.
echo  Tip: For best results, use the latest NVIDIA driver and CUDA Toolkit.
echo ====================================================
echo.

pause
endlocal