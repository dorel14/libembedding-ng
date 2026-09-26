<#
.SYNOPSIS
    Installe le runtime libcurl (DLL) requis par libembedding sur Windows.

.DESCRIPTION
    third_party/curl ne contient que les en-tetes et la bibliotheque
    d'import (lib/libcurl.dll.a, format GNU). Le lien passe, mais
    libembedding.dll comme chaque executable de test importent
    libcurl-x64.dll au demarrage : sans elle le processus meurt avec
    0xC0000135 (STATUS_DLL_NOT_FOUND), que ctest rapporte trompeusement
    comme un "Timeout".

    Ce script telecharge l'archive officielle curl-for-win correspondant a
    la version des en-tetes deja presents (LIBCURL_VERSION dans
    include/curl/curlver.h) et n'en extrait que bin/libcurl-x64.dll, dans
    third_party/curl/bin/ (emplacement canonique). Aucun binaire n'est
    versionne : le .gitignore ignore *.dll.

.PARAMETER CurlVersion
    Version de curl a installer. Par defaut, celle des en-tetes du depot.

.PARAMETER Force
    Retelecharge meme si la DLL est deja presente.

.PARAMETER ExpectedSha256
    SHA-256 attendu pour l'archive. Vide par defaut (le hash calcule est
    affiche et trace dans third_party/curl/bin/.libcurl-source.txt).

.EXAMPLE
    pwsh -File scripts/fetch_windows_libcurl.ps1

.NOTES
    Auteur: David Orel
    SPDX-License-Identifier: MIT
#>

[CmdletBinding()]
param(
    [string]$CurlVersion = '',
    [switch]$Force,
    [string]$ExpectedSha256 = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Windows PowerShell 5.1 needs TLS 1.2 explicitly for HTTPS downloads
[Net.ServicePointManager]::SecurityProtocol = `
    [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$repoRoot = Split-Path -Parent $PSScriptRoot
$headerFile = Join-Path $repoRoot 'third_party\curl\include\curl\curlver.h'
$importLib = Join-Path $repoRoot 'third_party\curl\lib\libcurl.dll.a'
$destDir = Join-Path $repoRoot 'third_party\curl\bin'
$destDll = Join-Path $destDir 'libcurl-x64.dll'
$provenance = Join-Path $destDir '.libcurl-source.txt'

function Write-Step { param([string]$Message) Write-Host "==> $Message" }
function Write-Info { param([string]$Message) Write-Host "    $Message" }
function Write-Warn { param([string]$Message) Write-Warning $Message }

# SHA-256 des archives curl-for-win verifiees (telechargement manuel puis
# controle du hash). Une archive listee ici doit correspondre exactement ;
# une archive absente est signalee mais acceptee (bump de version des
# en-tetes), son hash etant alors trace dans .libcurl-source.txt.
$KnownArchives = @{
    'curl-8.21.0_1-win64-mingw.zip' = '157068447d5b0b178dcc650f29d4746049fa4c7cc12db5f2bc050c0b84e48e7a'
}

function Get-HeaderCurlVersion {
    if (-not (Test-Path -LiteralPath $headerFile)) {
        throw "En-tetes libcurl introuvables : $headerFile"
    }
    $match = Select-String -LiteralPath $headerFile -Pattern 'define\s+LIBCURL_VERSION\s+"([^"]+)"' |
        Select-Object -First 1
    if (-not $match) {
        throw "LIBCURL_VERSION introuvable dans $headerFile"
    }
    return $match.Matches[0].Groups[1].Value
}

# The import library records the DLL name it binds to. Anything else than
# libcurl-x64.dll means the archive and the import library do not match.
function Get-ExpectedDllName {
    if (-not (Test-Path -LiteralPath $importLib)) {
        return 'libcurl-x64.dll'
    }
    $bytes = [IO.File]::ReadAllBytes($importLib)
    $text = [Text.Encoding]::ASCII.GetString($bytes)
    $found = [regex]::Matches($text, 'libcurl[a-z0-9\-\.]*\.dll') |
        ForEach-Object { $_.Value } |
        Select-Object -Unique
    if ($found) {
        return ($found | Select-Object -First 1)
    }
    return 'libcurl-x64.dll'
}

function Test-Url {
    param([string]$Uri)
    try {
        $response = Invoke-WebRequest -Uri $Uri -Method Head -UseBasicParsing
        return ($response.StatusCode -eq 200)
    } catch {
        return $false
    }
}

if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
    throw "Hote ARM64 : non gere (curl-for-win publie win64a, pas win64). Utilisez un hote x64 ou une installation libcurl propre a ARM64."
}

$version = if ($CurlVersion) { $CurlVersion } else { Get-HeaderCurlVersion }
$expectedDll = Get-ExpectedDllName

Write-Step "libcurl $version (runtime x64)"

if ((Test-Path -LiteralPath $destDll) -and -not $Force) {
    Write-Info "Deja installe : $destDll (utilisez -Force pour retelecharger)"
    exit 0
}

if ($expectedDll -ne 'libcurl-x64.dll') {
    throw "La bibliotheque d'import attend '$expectedDll' mais cet script n'installe que 'libcurl-x64.dll'. Verifiez third_party\curl\lib."
}

# The build suffix (_1, _2, ...) is not derivable from the headers: probe it.
$archive = $null
$archiveUrl = $null
foreach ($build in 1..3) {
    $candidate = "https://curl.se/windows/dl-${version}_${build}/curl-${version}_${build}-win64-mingw.zip"
    Write-Info "Test $candidate"
    if (Test-Url -Uri $candidate) {
        $archiveUrl = $candidate
        $archive = Split-Path -Leaf $candidate
        Write-Step "Archive $archive"
        break
    }
}

if (-not $archiveUrl) {
    throw "Aucune archive curl-for-win trouvee pour la version $version. Places manuellement la DLL dans $destDir (nom attendu : $expectedDll) ou utilisez -CurlVersion <version>."
}

$tmpRoot = Join-Path ([IO.Path]::GetTempPath()) ("libcurl-" + [Guid]::NewGuid().ToString('N'))
$zipPath = Join-Path $tmpRoot $archive
New-Item -ItemType Directory -Path $tmpRoot -Force | Out-Null

try {
    Write-Step "Telechargement"
    $progress = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    try {
        Invoke-WebRequest -Uri $archiveUrl -OutFile $zipPath -UseBasicParsing
    } finally {
        $ProgressPreference = $progress
    }

    $hash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Info "SHA-256 $hash"

    $pinned = if ($KnownArchives.ContainsKey($archive)) { $KnownArchives[$archive] } else { '' }
    if ($ExpectedSha256) {
        if ($hash -ne $ExpectedSha256.ToLowerInvariant()) {
            throw "SHA-256 inattendu : $hash (attendu $ExpectedSha256)"
        }
    } elseif ($pinned) {
        if ($hash -ne $pinned) {
            throw "SHA-256 inattendu pour $archive : $hash (attendu $pinned). L'archive publiee a change, verifiez avant de continuer."
        }
        Write-Info "Hash conforme a l'archive epinglee"
    } else {
        Write-Warn "$archive n'est pas epingle dans fetch_windows_libcurl.ps1 : verifiez le hash $hash avant de le committer dans .libcurl-source.txt"
    }

    Write-Step "Extraction de $expectedDll"
    Expand-Archive -LiteralPath $zipPath -DestinationPath $tmpRoot -Force

    $packaged = Get-ChildItem -Path $tmpRoot -Recurse -File -Filter $expectedDll |
        Select-Object -First 1
    if (-not $packaged) {
        $layout = (Get-ChildItem -Path $tmpRoot -Recurse -File | ForEach-Object { $_.Name }) -join ', '
        throw "$expectedDll absent de l'archive. Contenu : $layout"
    }

    New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    Copy-Item -LiteralPath $packaged.FullName -Destination $destDll -Force

    $size = (Get-Item -LiteralPath $destDll).Length
    if ($size -le 0) {
        throw "DLL extraite vide : $destDll"
    }

    @(
        "source   : $archiveUrl"
        "version  : $version"
        "sha256   : $hash"
        "pinned   : $(if ($pinned) { 'oui' } else { 'non' })"
        "file     : $expectedDll ($size octets)"
        "date     : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss K')"
    ) | Set-Content -LiteralPath $provenance -Encoding ASCII

    Write-Step "OK : $destDll ($size octets)"
    Write-Info "Provenance : $provenance"
} finally {
    Remove-Item -LiteralPath $tmpRoot -Recurse -Force -ErrorAction SilentlyContinue
}
