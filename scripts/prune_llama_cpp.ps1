<#
.SYNOPSIS
    Elague le snapshot llama.cpp vendorise dans third_party/.

.DESCRIPTION
    third_party/llama.cpp est une copie de l'amont (tag bNNNN), versionnee
    comme des fichiers ordinaires : il n'y a pas de sous-module, donc une
    mise a jour consiste a remplacer le dossier et tout revient, y compris
    les ~200 scripts Python, les 77 Mo de fixtures ggml-vocab-*.gguf et les
    docs/images qui ne servent jamais au build de libembedding.

    Ce script retire ce superflu. Il est idempotent et ne touche jamais ce
    que la cible llama/ggml requiert : CMakeLists.txt, LICENSE, AUTHORS,
    README.md, cmake/, ggml/, include/, src/, licenses/, vendor/.

    Regle de declencheur : lancer ce script APRES chaque remplacement du
    dossier third_party/llama.cpp, avant de commiter.

    Cote build, les options qui pointent vers les dossiers retires sont
    epinglees a OFF dans le CMakeLists.txt racine (LLAMA_BUILD_COMMON,
    LLAMA_BUILD_TOOLS, LLAMA_BUILD_APP, LLAMA_BUILD_MTMD, GGML_BUILD_TESTS,
    GGML_BUILD_EXAMPLES, GGML_OPENCL, GGML_WEBGPU, GGML_VIRTGPU).

.PARAMETER Root
    Racine du depot. Par defaut, le parent de ce script.

.PARAMETER DryRun
    Liste ce qui serait supprime sans toucher au disque.

.EXAMPLE
    pwsh -File scripts/prune_llama_cpp.ps1 -DryRun

.EXAMPLE
    pwsh -File scripts/prune_llama_cpp.ps1

.NOTES
    Auteur: David Orel
    SPDX-License-Identifier: MIT
#>

[CmdletBinding()]
param(
    [string]$Root = '',
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Step { param([string]$Message) Write-Host "==> $Message" }
function Write-Info { param([string]$Message) Write-Host "    $Message" }
function Write-Warn { param([string]$Message) Write-Warning $Message }

if ([string]::IsNullOrWhiteSpace($Root)) {
    $Root = Split-Path -Parent $PSScriptRoot
}
$llamaDir = Join-Path $Root 'third_party\llama.cpp'

if (-not (Test-Path -LiteralPath (Join-Path $llamaDir 'CMakeLists.txt'))) {
    throw "third_party/llama.cpp/CMakeLists.txt introuvable sous '$Root'. Lancez le script depuis la racine du depot."
}

# Repertoires amenes par l'amont mais jamais references par le build.
$dirs = @(
    '.devops', '.gemini', '.github', '.pi',
    'app', 'benches', 'ci', 'common', 'conversion', 'docs', 'examples',
    'grammars', 'gguf-py', 'media', 'models', 'pocs', 'scripts', 'skills',
    'tests', 'tools'
)

# Fichiers amenes par l'amont, hors outillage de build conserve.
$files = @(
    '.clang-format', '.clang-tidy', '.dockerignore', '.ecrc', '.editorconfig',
    '.flake8', '.gitignore', '.gitmodules', '.pre-commit-config.yaml',
    'AGENTS.md', 'CLAUDE.md', 'CMakePresets.json', 'CODEOWNERS',
    'CONTRIBUTING.md', 'Makefile', 'SECURITY.md',
    'build-xcframework.sh', 'flake.nix', 'mypy.ini', 'pyproject.toml',
    'pyrightconfig.json', 'requirements.txt', 'ty.toml',
    'convert_hf_to_gguf.py', 'convert_hf_to_gguf_update.py',
    'convert_llama_ggml_to_gguf.py', 'convert_lora_to_gguf.py'
)

# Conserves : generateurs appeles par CMake pour des backends optionnels
# (GGML_WEBGPU / GGML_OPENCL, epingles a OFF). Les supprimer casse la
# configure si quelqu'un active ces backends.
$keptCodegen = @(
    'ggml\src\ggml-webgpu\wgsl-shaders\embed_wgsl.py',
    'ggml\src\ggml-opencl\kernels\embed_kernel.py'
)

Write-Step "Elagage de $llamaDir"
if ($DryRun) { Write-Info 'DryRun : aucune suppression.' }

$removedDirs = 0
$removedFiles = 0

foreach ($name in $dirs) {
    $path = Join-Path $llamaDir $name
    if (-not (Test-Path -LiteralPath $path)) { continue }
    if ($DryRun) {
        Write-Info "supprimerait  $name/"
        $removedDirs++
        continue
    }
    Remove-Item -LiteralPath $path -Recurse -Force
    Write-Info "supprime  $name/"
    $removedDirs++
}

if (Test-Path -LiteralPath (Join-Path $llamaDir 'requirements')) {
    $path = Join-Path $llamaDir 'requirements'
    if ($DryRun) {
        Write-Info 'supprimerait  requirements/'
        $removedDirs++
    } else {
        Remove-Item -LiteralPath $path -Recurse -Force
        Write-Info 'supprime  requirements/'
        $removedDirs++
    }
}

foreach ($name in $files) {
    $path = Join-Path $llamaDir $name
    if (-not (Test-Path -LiteralPath $path)) { continue }
    if ($DryRun) {
        Write-Info "supprimerait  $name"
        $removedFiles++
        continue
    }
    Remove-Item -LiteralPath $path -Force
    Write-Info "supprime  $name"
    $removedFiles++
}

foreach ($name in $keptCodegen) {
    $path = Join-Path $llamaDir $name
    if (-not (Test-Path -LiteralPath $path)) {
        Write-Warn "attendu mais absent : $name (regenere a la mise a jour ?)"
    }
}

Write-Info "$removedDirs repertoire(s), $removedFiles fichier(s) supprime(s)."
Write-Step 'Verifiez la configuration avant de commiter :'
Write-Info 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLIBEMBEDDING_BUILD_TESTS=ON'
