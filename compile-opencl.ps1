# PowerShell version of compile-opencl.bat
$ErrorActionPreference = "Stop"

# First loop for architectures
$archs1 = @("gfx1011", "gfx1012", "gfx1030", "gfx1031", "gfx1032", "gfx1034", "gfx906")
foreach ($arch in $archs1) {
    Write-Host "Building for architecture: $arch"
    try {
        & rga --O3 -s opencl -c $arch --OpenCLoption "-cl-finite-math-only -cl-mad-enable " -b plugins\opencl\resources\bin\vecno-opencl plugins\opencl\resources\heavy_hash.cl -D __FORCE_AMD_V_DOT8_U32_U4=1 -D OPENCL_PLATFORM_AMD -D OFFLINE 2>&1 | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "rga failed for $arch with exit code $LASTEXITCODE"
        }
    }
    catch {
        Write-Host "Error building for $arch : $_"
        exit 1
    }
}

# Second loop for architecture: gfx1010
$archs2 = @("gfx1010")
foreach ($arch in $archs2) {
    Write-Host "Building for architecture: $arch"
    try {
        & rga --O3 -s opencl -c $arch --OpenCLoption "-cl-finite-math-only -cl-mad-enable " -b plugins\opencl\resources\bin\vecno-opencl plugins\opencl\resources\heavy_hash.cl -D OPENCL_PLATFORM_AMD 2>&1 | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "rga failed for $arch with exit code $LASTEXITCODE"
        }
    }
    catch {
        Write-Host "Error building for $arch : $_"
        exit 1
    }
}

# Third loop for newer architectures: gfx1100, gfx1101, gfx1201
$archs3 = @("gfx1100", "gfx1101", "gfx1201")
foreach ($arch in $archs3) {
    Write-Host "Building for architecture: $arch"
    try {
        & rga --O3 -s opencl -c $arch --OpenCLoption "-cl-mad-enable -cl-finite-math-only " -b plugins\opencl\resources\bin\vecno-opencl plugins\opencl\resources\heavy_hash.cl -D OPENCL_PLATFORM_AMD 2>&1 | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "rga failed for $arch with exit code $LASTEXITCODE"
        }
    }
    catch {
        Write-Host "Error building for $arch : $_"
        exit 1
    }
}

Write-Host "Build completed successfully."