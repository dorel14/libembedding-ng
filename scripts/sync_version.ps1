# One-time version sync script
# Temp file - mechanical text replacement, no code logic change
$ErrorActionPreference = "SilentlyContinue"

$files = @()
Get-ChildItem -Path "include" -Recurse -Filter "*.h" -ErrorAction SilentlyContinue | ForEach-Object { $files += $_.FullName }
Get-ChildItem -Path "include" -Recurse -Filter "*.hpp" -ErrorAction SilentlyContinue | ForEach-Object { $files += $_.FullName }

$updated = 0
foreach ($f in $files) {
  $content = Get-Content $f -Raw -ErrorAction SilentlyContinue
  if ($content -match "Version: 1\.4\.0") {
    $new = $content -replace "Version: 1\.4\.0", "Version: 1.6.0"
    if ($new -ne $content) {
      $enc = [System.Text.UTF8Encoding]::new($false)
      [System.IO.File]::WriteAllText($f, $new, $enc)
      Write-Output "Updated: $f"
      $updated++
    }
  }
}
Write-Output "Total updated: $updated"
