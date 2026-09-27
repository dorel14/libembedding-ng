#!/usr/bin/env pwsh
# run_tests.ps1 — Run libembedding tests on Windows (PowerShell)
# Usage: .\run_tests.ps1 [-Integration] [-Jobs N]

param(
    [switch]$Integration = $false,
    [int]$Jobs = 0
)

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot
$BuildDir = Join-Path $ProjectDir "build"

if ($Jobs -eq 0) {
    $Jobs = [Environment]::ProcessorCount
}

Write-Host "=== libembedding tests (Windows) ===" -ForegroundColor Cyan

# Build first
$BuildConfig = "Debug"
$extraFlags = ""
if ($Integration) {
    $extraFlags = "-DLIBEMBEDDING_INTEGRATION_TESTS=ON"
    Write-Host "  Integration tests: ENABLED (requires network)" -ForegroundColor Yellow
} else {
    Write-Host "  Integration tests: disabled (pass -Integration to enable)"
}

if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
}

# The libcurl runtime DLL is not vendored; CMake fails at configure time
# without it. Install it on demand so a fresh clone tests in one command.
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
    -DCMAKE_BUILD_TYPE=$BuildConfig `
    -DLIBEMBEDDING_BUILD_TESTS=ON `
    -DLIBEMBEDDING_BUILD_EXAMPLES=OFF `
    $extraFlags

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed." -ForegroundColor Red
    exit 1
}

# --config is required by multi-config generators (MSBuild): the build
# otherwise lands in the generator default instead of $BuildConfig.
cmake --build $BuildDir --parallel $Jobs --config $BuildConfig

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "--- Running tests ---" -ForegroundColor Cyan
cd $BuildDir
# "-j$Jobs" must stay quoted: an unquoted -j$Jobs is passed to ctest as a
# literal parameter token ("-j$Jobs") and ctest rejects the value. -C is
# mandatory for multi-config generators (MSBuild).
ctest --output-on-failure "-j$Jobs" -C $BuildConfig

if ($LASTEXITCODE -ne 0) {
    Write-Host "Some tests failed." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== All tests passed ===" -ForegroundColor Green