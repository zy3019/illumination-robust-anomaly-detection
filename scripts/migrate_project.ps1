param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('CopyAndVerify', 'Cutover')]
    [string]$Phase
)
$ErrorActionPreference = 'Stop'
$sourceRoot = 'C:\Users\LENOVO\Documents\ChatGPT\111'
$destinationRoot = 'D:\ChatGPT\111'
$backupRoot = 'C:\Users\LENOVO\Documents\ChatGPT\111.migration-backup-20261006'
$manifestPath = 'D:\ChatGPT\111-migration-20261006.json'
$copyLog = 'D:\ChatGPT\111-migration-copy-20261006.log'

function Assert-ExactDirectory([string]$Path, [string]$Expected, [bool]$AllowJunction = $false) {
    $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).ProviderPath.TrimEnd('\')
    if ($resolved -ne $Expected.TrimEnd('\')) { throw "Unexpected resolved path: $resolved" }
    $item = Get-Item -LiteralPath $Path -Force
    if (-not $item.PSIsContainer) { throw "Expected directory: $Path" }
    if (-not $AllowJunction -and ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Refusing reparse-point directory: $Path"
    }
}

function Get-Inventory([string]$Root) {
    $items = @(Get-ChildItem -LiteralPath $Root -Force -Recurse -ErrorAction Stop)
    if (@($items | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) {
        throw "Unexpected reparse point inside $Root"
    }
    return @($items | Where-Object { -not $_.PSIsContainer } | Sort-Object FullName)
}

if ($Phase -eq 'CopyAndVerify') {
    Assert-ExactDirectory $sourceRoot 'C:\Users\LENOVO\Documents\ChatGPT\111'
    if (Test-Path -LiteralPath $destinationRoot) { throw 'Destination already exists; refusing to merge' }
    if (Test-Path -LiteralPath $backupRoot) { throw 'Backup path already exists' }
    $sourceFiles = Get-Inventory $sourceRoot
    $totalBytes = ($sourceFiles | Measure-Object Length -Sum).Sum
    if ((Get-PSDrive D).Free -lt ($totalBytes + 1GB)) { throw 'Insufficient destination space' }
    New-Item -ItemType Directory -Path $destinationRoot -Force | Out-Null
    Assert-ExactDirectory $destinationRoot 'D:\ChatGPT\111'
    Write-Output "Copying $($sourceFiles.Count) files, $totalBytes bytes"
    & robocopy.exe $sourceRoot $destinationRoot /E /COPY:DAT /DCOPY:DAT /XJ /R:1 /W:1 /MT:8 /NFL /NDL /NP "/LOG:$copyLog"
    $copyExit = $LASTEXITCODE
    if ($copyExit -ge 8) { throw "Robocopy failed ($copyExit); original retained" }
    Write-Output "Copy finished (robocopy=$copyExit); verifying all SHA-256 hashes"
    $sourceFiles = Get-Inventory $sourceRoot
    $destinationFiles = Get-Inventory $destinationRoot
    if ($sourceFiles.Count -ne $destinationFiles.Count) { throw 'File-count mismatch' }
    $entries = [System.Collections.Generic.List[object]]::new()
    $count = 0
    foreach ($file in $sourceFiles) {
        $relative = $file.FullName.Substring($sourceRoot.Length + 1)
        $target = Join-Path $destinationRoot $relative
        $sourceHash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
        $targetFile = Get-Item -LiteralPath $target
        if ($file.Length -ne $targetFile.Length) { throw "Size mismatch: $relative" }
        $targetHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
        if ($sourceHash -ne $targetHash) { throw "Hash mismatch: $relative" }
        $entries.Add([pscustomobject]@{ path=$relative; bytes=$file.Length; sha256=$sourceHash })
        $count++
        if ($count % 2500 -eq 0) { Write-Output "Verified $count / $($sourceFiles.Count) files" }
    }
    $sourceDirs = @(Get-ChildItem -LiteralPath $sourceRoot -Directory -Force -Recurse | ForEach-Object { $_.FullName.Substring($sourceRoot.Length + 1) })
    $destDirs = @(Get-ChildItem -LiteralPath $destinationRoot -Directory -Force -Recurse | ForEach-Object { $_.FullName.Substring($destinationRoot.Length + 1) })
    if (Compare-Object $sourceDirs $destDirs) { throw 'Directory-tree mismatch' }
    [pscustomobject]@{ source=$sourceRoot; destination=$destinationRoot; status='copy_sha256_verified';
        robocopy_exit=$copyExit; file_count=$entries.Count; bytes=$totalBytes; files=$entries } |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding utf8
    Write-Output "Verified complete copy: $($entries.Count) files, $totalBytes bytes"
    exit 0
}

# Copy verification is mandatory before switching the original path.
Assert-ExactDirectory $sourceRoot 'C:\Users\LENOVO\Documents\ChatGPT\111'
Assert-ExactDirectory $destinationRoot 'D:\ChatGPT\111'
if (Test-Path -LiteralPath $backupRoot) { throw 'Backup path already exists' }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.status -ne 'copy_sha256_verified' -or $manifest.source -ne $sourceRoot -or $manifest.destination -ne $destinationRoot) {
    throw 'Missing or invalid verification manifest'
}
$availableBefore = (Get-PSDrive C).Free
# These exact paths have been resolved and checked above. The backup is a
# same-volume rename, and is retained until the relocated runtime is verified.
Move-Item -LiteralPath $sourceRoot -Destination $backupRoot
Assert-ExactDirectory $backupRoot 'C:\Users\LENOVO\Documents\ChatGPT\111.migration-backup-20261006'
New-Item -ItemType Junction -Path $sourceRoot -Target $destinationRoot | Out-Null
$junction = Get-Item -LiteralPath $sourceRoot -Force
if ($junction.LinkType -ne 'Junction' -or @($junction.Target)[0] -ne $destinationRoot) { throw 'Compatibility junction is incorrect' }

# Retarget the audit environment's explicit dependency path. Historical records
# and launchers retain their original paths, supported by the compatibility link.
$auditPth = Join-Path $destinationRoot '.venv-audit\Lib\site-packages\project_runtime.pth'
Set-Content -LiteralPath $auditPth -Value 'D:\ChatGPT\111\.venv\Lib\site-packages' -Encoding ascii
Set-Location -LiteralPath $destinationRoot
& 'D:\ChatGPT\111\.venv\Scripts\python.exe' -B -c "import sys,torch; from pathlib import Path; assert Path(sys.prefix).resolve()==Path(r'D:\ChatGPT\111\.venv').resolve(); assert torch.cuda.is_available(); x=torch.tensor([2.0],device='cuda'); assert (x*x).item()==4.0; print('Main runtime and CUDA passed:',sys.prefix)"
if ($LASTEXITCODE -ne 0) { throw 'Relocated main runtime check failed; C backup retained' }
& 'D:\ChatGPT\111\.venv\Scripts\python.exe' -B -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Relocated tests failed; C backup retained' }
$env:MPLCONFIGDIR = 'D:\ChatGPT\111\data\cache\matplotlib'
& 'D:\ChatGPT\111\.venv-audit\Scripts\python.exe' -B -c "import sys,torch,anomalib; from pathlib import Path; from anomalib.models.image.patchcore.torch_model import PatchcoreModel; assert Path(torch.__file__).resolve().is_relative_to(Path(r'D:\ChatGPT\111').resolve()); assert anomalib.__version__=='2.2.0'; print('Audit runtime passed:',sys.prefix)"
if ($LASTEXITCODE -ne 0) { throw 'Relocated audit runtime check failed; C backup retained' }

# Re-hash the renamed originals to detect edits after the copy. Also verify the
# destination again before deleting the exact, bounded backup tree.
$backupFiles = Get-Inventory $backupRoot
if ($backupFiles.Count -ne $manifest.file_count) { throw 'Original files changed after copying; backup retained' }
$count = 0
foreach ($entry in $manifest.files) {
    $oldFile = Join-Path $backupRoot $entry.path
    $newFile = Join-Path $destinationRoot $entry.path
    if ((Get-FileHash -LiteralPath $oldFile -Algorithm SHA256).Hash -ne $entry.sha256) {
        throw "Original changed after copy: $($entry.path); backup retained"
    }
    if ($entry.path -ne '.venv-audit\Lib\site-packages\project_runtime.pth') {
        if ((Get-FileHash -LiteralPath $newFile -Algorithm SHA256).Hash -ne $entry.sha256) {
            throw "Destination changed unexpectedly: $($entry.path); backup retained"
        }
    }
    $count++
    if ($count % 5000 -eq 0) { Write-Output "Pre-cleanup verified $count / $($manifest.file_count) files" }
}
Assert-ExactDirectory $backupRoot 'C:\Users\LENOVO\Documents\ChatGPT\111.migration-backup-20261006'
if (-not $backupRoot.StartsWith('C:\Users\LENOVO\Documents\ChatGPT\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Backup escaped approved parent' }
Assert-ExactDirectory $destinationRoot 'D:\ChatGPT\111'
Remove-Item -LiteralPath $backupRoot -Recurse -Force
if (Test-Path -LiteralPath $backupRoot) { throw 'Backup cleanup incomplete' }
$availableAfter = (Get-PSDrive C).Free
$receipt = [pscustomobject]@{ status='complete'; source_compatibility_junction=$sourceRoot; destination=$destinationRoot;
    verified_file_count=$manifest.file_count; verified_bytes=$manifest.bytes; c_free_before=$availableBefore;
    c_free_after=$availableAfter; observed_c_free_increase=$availableAfter-$availableBefore;
    tests='7 unit tests; main CUDA; Anomalib import'; adjusted_path=$auditPth; manifest=$manifestPath;
    note='Base Python remains C:\Python314. Old absolute launchers remain usable through the junction.' }
$receipt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath 'D:\ChatGPT\111\records\migration-20261006.json' -Encoding utf8
$receipt | ConvertTo-Json -Depth 4 | Write-Output
