$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$classifierCheckpoint = Join-Path $projectRoot "results/efficientnet_b0_gan_augmented/best_efficientnet_b0.pt"
$sampleDirectory = Join-Path $projectRoot "results/gan_individual_samples"
$approvalFile = Join-Path $sampleDirectory "TRAINING_APPROVED.txt"
if (Test-Path -LiteralPath $classifierCheckpoint) {
    throw "Refusing to overwrite existing checkpoint: $classifierCheckpoint"
}
if (-not (Test-Path -LiteralPath $approvalFile)) {
    throw "GAN samples are not approved for training. Generate and inspect results/gan_individual_samples first, then create $approvalFile after verifying the class labels and visual quality."
}

$pythonExe = $env:ROAD_DAMAGE_PYTHON
if (-not $pythonExe) {
    $localPython = Join-Path $env:LOCALAPPDATA "Python/bin/python.exe"
    if (Test-Path -LiteralPath $localPython) {
        $pythonExe = $localPython
    }
    else {
        $pythonCommand = Get-Command python -ErrorAction Stop
        $pythonExe = $pythonCommand.Source
    }
}
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python executable not found: $pythonExe. Set ROAD_DAMAGE_PYTHON to your Python executable."
}

Push-Location $projectRoot
try {
    $samples = Get-ChildItem -LiteralPath $sampleDirectory -Recurse -File -Include *.png, *.jpg, *.jpeg
    if (-not $samples) {
        throw "No approved GAN training samples found under $sampleDirectory"
    }

    $env:ROAD_DAMAGE_USE_GAN_AUGMENT = "1"
    & $pythonExe src/39_train_efficientnet.py
    if ($LASTEXITCODE -ne 0) {
        throw "GAN-augmented EfficientNet training failed with exit code $LASTEXITCODE"
    }
}
finally {
    Remove-Item Env:\ROAD_DAMAGE_USE_GAN_AUGMENT -ErrorAction SilentlyContinue
    Pop-Location
}

Write-Host "GAN-augmented EfficientNet training completed. Check results/efficientnet_b0_gan_augmented/ before using it in the dashboard."
