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

cmake -S $ProjectDir -B $BuildDir `
    -DCMAKE_BUILD_TYPE=Debug `
    -DLIBEMBEDDING_BUILD_TESTS=ON `
    -DLIBEMBEDDING_BUILD_EXAMPLES=OFF `
    $extraFlags

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed." -ForegroundColor Red
    exit 1
}

cmake --build $BuildDir --parallel $Jobs

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "--- Running tests ---" -ForegroundColor Cyan
cd $BuildDir
ctest --output-on-failure -j$Jobs

if ($LASTEXITCODE -ne 0) {
    Write-Host "Some tests failed." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== All tests passed ===" -ForegroundColor Green