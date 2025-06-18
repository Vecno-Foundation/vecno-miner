$ErrorActionPreference = "Stop"

# Define architectures
$archs_experimental = @("gfx1011", "gfx1012", "gfx1030", "gfx1031", "gfx1032", "gfx1034", "gfx906")
$archs_standard = @("gfx1010", "gfx1100", "gfx1101", "gfx1201")

# Common compiler options
$commonOptions = "-cl-finite-math-only -cl-mad-enable -cl-std=CL2.0 -D AMD_ACCELERATED_PROCESSING -D OPENCL_PLATFORM_AMD"
$outputDir = "plugins\opencl\resources\bin"
$sourceFile = "plugins\opencl\resources\heavy_hash.cl"

# Create output directory
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

# Build experimental binaries for older architectures
foreach ($arch in $archs_experimental) {
    Write-Host "Building experimental binary for architecture: $arch (packed matrix)"
    $outputFile = "$outputDir\vecno-opencl.bin"
    Write-Host "Output file: $outputFile"
    try {
        $options = "$commonOptions -D __${arch}__ -D __FORCE_AMD_V_DOT8_U32_U4=1 -D EXPERIMENTAL_AMD -D OFFLINE"
        & rga --O3 -s opencl -c $arch --OpenCLoption $options -b $outputFile $sourceFile 2>&1 | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "rga failed for $arch (experimental) with exit code $LASTEXITCODE"
        }
    }
    catch {
        Write-Host "Error building experimental binary for $arch : $_"
        exit 1
    }
}

# Build standard binaries for newer architectures
foreach ($arch in $archs_standard) {
    Write-Host "Building standard binary for architecture: $arch (unpacked matrix)"
    $outputFile = "$outputDir\vecno-opencl.bin"
    Write-Host "Output file: $outputFile"
    try {
        $options = "$commonOptions -D __${arch}__"
        & rga --O3 -s opencl -c $arch --OpenCLoption $options -b $outputFile $sourceFile 2>&1 | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "rga failed for $arch with exit code $LASTEXITCODE"
        }
    }
    catch {
        Write-Host "Error building standard binary for $arch : $_"
        exit 1
    }
}

Write-Host "Build completed successfully."