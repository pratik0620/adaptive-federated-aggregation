param (
    [string]$ConfigPath = "configs/padchest_partial.yaml",
    [switch]$SkipExtraction,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  PadChest Client 03 - Automated Data Prep and Training Pipeline" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$PythonExe = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python.exe"
}

# -----------------------------------------------------------------------------
# STAGE 1: Environment & Dependencies
# -----------------------------------------------------------------------------
Write-Host "`n>>> [STAGE 1/6] Verifying Python Environment and Core Dependencies..." -ForegroundColor Yellow
& $PythonExe src/verify_environment.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Stage 1 Failed: Environment verification did not pass." -ForegroundColor Red
    exit 1
}

# -----------------------------------------------------------------------------
# STAGE 2: Archives and Manifests Audit
# -----------------------------------------------------------------------------
Write-Host "`n>>> [STAGE 2/6] Auditing Archives (0.zip, 1.zip) and Manifests..." -ForegroundColor Yellow

$ArchivesDir = "data/raw/PadChest/archives"
$ManifestsDir = "data/raw/PadChest/archive_manifests"
$MetadataDir = "data/raw/PadChest/metadata"
$ImagesDir = "data/raw/PadChest/images"

New-Item -ItemType Directory -Force -Path $ArchivesDir, $ManifestsDir, $MetadataDir, $ImagesDir | Out-Null

$File0Zip = Join-Path $ArchivesDir "0.zip"
$File1Zip = Join-Path $ArchivesDir "1.zip"
$File0Manifest = Join-Path $ManifestsDir "0.zip.unzip-l.txt"
$File1Manifest = Join-Path $ManifestsDir "1.zip.unzip-l.txt"

$MissingArchives = @()
if (-not (Test-Path $File0Zip)) {
    $MissingArchives += "Archive: $File0Zip (Required: 0.zip, ~20.35 GB)"
}
if (-not (Test-Path $File1Zip)) {
    $MissingArchives += "Archive: $File1Zip (Required: 1.zip, ~20.49 GB)"
}
if (-not (Test-Path $File0Manifest)) {
    $MissingArchives += "Manifest: $File0Manifest (Required: 0.zip.unzip-l.txt)"
}
if (-not (Test-Path $File1Manifest)) {
    $MissingArchives += "Manifest: $File1Manifest (Required: 1.zip.unzip-l.txt)"
}

if ($MissingArchives.Count -gt 0) {
    Write-Host "`n[STOPPING - MISSING ARCHIVE FILES]" -ForegroundColor Red
    Write-Host "Please ensure both archives and manifests are present:" -ForegroundColor Red
    foreach ($m in $MissingArchives) {
        Write-Host "  - $m" -ForegroundColor Yellow
    }
    exit 2
}

# Audit 0.zip and 1.zip against manifests without extracting
& $PythonExe src/verify_and_extract_padchest.py --audit_only
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Stage 2 Failed: Archive audit against manifests did not pass." -ForegroundColor Red
    exit 2
}

# -----------------------------------------------------------------------------
# STAGE 3: Extract and Verify Images
# -----------------------------------------------------------------------------
Write-Host "`n>>> [STAGE 3/6] Extracting and Verifying Image Archives..." -ForegroundColor Yellow
if ($SkipExtraction) {
    Write-Host "  (SkipExtraction flag set. Running image integrity verification only...)" -ForegroundColor Gray
    & $PythonExe src/verify_and_extract_padchest.py --verify_only
} else {
    & $PythonExe src/verify_and_extract_padchest.py --extract
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Stage 3 Failed: Archive extraction or image verification error." -ForegroundColor Red
    exit 3
}

# -----------------------------------------------------------------------------
# STAGE 4: Official Metadata Verification (Strict Gate)
# -----------------------------------------------------------------------------
Write-Host "`n>>> [STAGE 4/6] Verifying Official 160K Metadata File..." -ForegroundColor Yellow
$FileMetaGz = Join-Path $MetadataDir "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz"
$FileMetaCsv = Join-Path $MetadataDir "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"

$OfficialMetaPresent = (Test-Path $FileMetaGz) -or (Test-Path $FileMetaCsv)
if (-not $OfficialMetaPresent) {
    Write-Host "`n======================================================================" -ForegroundColor Red
    Write-Host "  [STOPPED BEFORE PREPROCESSING & TRAINING] OFFICIAL METADATA MISSING" -ForegroundColor Red
    Write-Host "======================================================================" -ForegroundColor Red
    Write-Host "Both archives (0.zip and 1.zip) are verified and extracted, BUT the official" -ForegroundColor Yellow
    Write-Host "160K PadChest metadata CSV is missing." -ForegroundColor Yellow
    Write-Host "`nRequired file:" -ForegroundColor White
    Write-Host "  data/raw/PadChest/metadata/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz" -ForegroundColor Cyan
    Write-Host "  (or uncompressed: PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv)" -ForegroundColor Cyan
    Write-Host "`nNOTE: The Kaggle sample CSV will NOT be substituted as it only contains 24 images." -ForegroundColor Yellow
    Write-Host "Please place the official metadata CSV in data/raw/PadChest/metadata/ and rerun.`n" -ForegroundColor Yellow
    exit 4
}

Write-Host "  [OK] Official 160K metadata verified." -ForegroundColor Green

# -----------------------------------------------------------------------------
# STAGE 5: Preprocessing & Preflight Dry-Run
# -----------------------------------------------------------------------------
Write-Host "`n>>> [STAGE 5/6] Preprocessing Metadata & Running Preflight Validation..." -ForegroundColor Yellow
& $PythonExe src/preprocess_padchest.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Stage 5 Failed: Preprocessing pipeline failed." -ForegroundColor Red
    exit 5
}

& $PythonExe src/train_padchest.py --config $ConfigPath --dry_run
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Stage 5 Failed: Preflight dry-run check failed." -ForegroundColor Red
    exit 5
}

if ($PreflightOnly) {
    Write-Host "`n[COMPLETED] Preflight validation finished successfully (PreflightOnly set). Halting before training." -ForegroundColor Green
    exit 0
}

# -----------------------------------------------------------------------------
# STAGE 6: Model Training, Federated Verification, and Packaging
# -----------------------------------------------------------------------------
Write-Host "`n>>> [STAGE 6/6] Starting ResNet-50 Binary Model Training (CPU-Optimized)..." -ForegroundColor Yellow
& $PythonExe src/train_padchest.py --config $ConfigPath
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Stage 6 Failed: Model training encountered an error." -ForegroundColor Red
    exit 6
}

Write-Host "`n>>> Validating Federated Checkpoint Compatibility..." -ForegroundColor Yellow
& $PythonExe src/verify_checkpoint_compatibility.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Checkpoint compatibility verification failed." -ForegroundColor Red
    exit 6
}

Write-Host "`n>>> Packaging Results for Federated Project Lead..." -ForegroundColor Yellow
& $PythonExe src/package_padchest_results.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Results packaging failed." -ForegroundColor Red
    exit 6
}

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host "  [SUCCESS] PadChest Client 03 Pipeline Finished Successfully!" -ForegroundColor Green
Write-Host "  Final Package: PadChest_Client03_Training_Results.zip" -ForegroundColor Green
Write-Host "======================================================================`n" -ForegroundColor Green
