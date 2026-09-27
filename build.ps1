#!/usr/bin/env pwsh
# build.ps1 — Build script for libembedding on Windows (PowerShell)
# Usage: .\build.ps1 [Debug|Release] [-Jobs N]

param(
    [string]$BuildType = "Release",
    [int]$Jobs = 0
)

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot
$BuildDir = Join-Path $ProjectDir "build"

if ($Jobs -eq 0) {
    $Jobs = [Environment]::ProcessorCount
}

Write-Host "=== libembedding build (Windows) ===" -ForegroundColor Cyan
Write-Host "  Build type:  $BuildType"
Write-Host "  Parallel:    $Jobs jobs"
Write-Host "  Build dir:   $BuildDir"
Write-Host ""

# Configure
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
}

# The libcurl runtime DLL is not vendored; CMake fails at configure time
# without it. Install it on demand so a fresh clone builds in one command.
$CurlRuntime = Join-Path $ProjectDir "third_party/curl/bin/libcurl-x64.dll"
if (-not (Test-Path $CurlRuntime)) {
    Write-Host "Installing the libcurl runtime (one-time)..." -ForegroundColor Cyan
    try {
        & (Join-Path $ProjectDir "scripts/fetch_windows_libcurl.ps1")
    } catch {
        Write-Host $_.Exception.Message -ForegroundColor Red
    }
    if (-not (Test-Path $CurlRuntime)) {
        Write-Host "Could not install the libcurl runtime. Re-run with" -ForegroundColor Red
        Write-Host "  -DLIBEMBEDDING_NO_DOWNLOAD=ON, or provide -DCURL_LIBRARY=<path>." -ForegroundColor Red
        exit 1
    }
}

cmake -S $ProjectDir -B $BuildDir `
    -DCMAKE_BUILD_TYPE=$BuildType `
    -DLIBEMBEDDING_BUILD_TESTS=ON `
    -DLIBEMBEDDING_BUILD_EXAMPLES=ON

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed." -ForegroundColor Red
    exit 1
}

# Build
cmake --build $BuildDir --parallel $Jobs

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Build complete ===" -ForegroundColor Green
Write-Host "  Tests:    $BuildDir/tests/"
Write-Host "  Examples: $BuildDir/examples/"