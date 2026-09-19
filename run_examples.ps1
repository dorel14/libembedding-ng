#!/usr/bin/env pwsh
# run_examples.ps1 — Run libembedding examples on Windows (PowerShell)
# Usage: .\run_examples.ps1 <example_name> [-Jobs N]

param(
    [string]$Example = "basic_embedding",
    [int]$Jobs = 0
)

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot
$BuildDir = Join-Path $ProjectDir "build"

if ($Jobs -eq 0) {
    $Jobs = [Environment]::ProcessorCount
}

Write-Host "=== libembedding examples (Windows) ===" -ForegroundColor Cyan

# Build
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
}

cmake -S $ProjectDir -B $BuildDir `
    -DCMAKE_BUILD_TYPE=Release `
    -DLIBEMBEDDING_BUILD_TESTS=OFF `
    -DLIBEMBEDDING_BUILD_EXAMPLES=ON

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed." -ForegroundColor Red
    exit 1
}

cmake --build $BuildDir --parallel $Jobs

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed." -ForegroundColor Red
    exit 1
}

$Binary = Join-Path $BuildDir "examples/$Example"
if (-not (Test-Path $Binary)) {
    Write-Host "Error: example '$Example' not found." -ForegroundColor Red
    Write-Host "Available examples:"
    Get-ChildItem (Join-Path $BuildDir "examples") -File | ForEach-Object { Write-Host "  $($_.Name)" }
    exit 1
}

Write-Host ""
Write-Host "--- Running: $Example ---" -ForegroundColor Cyan
Write-Host ""
& $Binary