param(
    [string]$FunctionName = "arc",
    [string]$Region = $(if ($env:AWS_REGION) { $env:AWS_REGION } else { "us-east-1" }),
    [string]$RedisUrl = $env:REDIS_URL,
    [string]$LambdaUrl = $env:LAMBDA_URL
)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

function Invoke-Checked([scriptblock]$Command) {
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code ${LASTEXITCODE}: $Command" }
}

foreach ($tool in @("docker", "aws", "git")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool is not on PATH." }
}
$bundle = Join-Path $RepoRoot "artifacts\model_bundle.joblib"
if (-not (Test-Path -LiteralPath $bundle)) { throw "Run the pipeline first: model bundle missing." }
if (Invoke-Checked { git -C $RepoRoot status --porcelain }) { throw "Commit your changes first: the image is tagged with the commit it was built from." }

$AccountId = (aws sts get-caller-identity --query Account --output text)
if (-not $AccountId) { throw "Configure the AWS CLI." }
$Registry = "$AccountId.dkr.ecr.$Region.amazonaws.com"
$Repository = "$Registry/$FunctionName"
$GitSha = Invoke-Checked { git -C $RepoRoot rev-parse --short HEAD }
$ImageLatest = "${Repository}:latest"
$ImageSha = "${Repository}:$GitSha"

Invoke-Checked { docker build --provenance=false -t $FunctionName -f (Join-Path $RepoRoot "Dockerfile.lambda") $RepoRoot }
Invoke-Checked { docker tag "${FunctionName}:latest" $ImageLatest }
Invoke-Checked { docker tag "${FunctionName}:latest" $ImageSha }
Invoke-Checked { aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $Registry }
Invoke-Checked { docker push $ImageLatest }
Invoke-Checked { docker push $ImageSha }

if ($RedisUrl) {
    $LambdaVars = @{}
    $existingJson = Invoke-Checked { aws lambda get-function-configuration --function-name $FunctionName --region $Region --query "Environment.Variables" --output json }
    $existing = $existingJson | ConvertFrom-Json
    if ($existing) {
        foreach ($entry in $existing.PSObject.Properties) { $LambdaVars[$entry.Name] = $entry.Value }
    }
    $LambdaVars["REDIS_URL"] = $RedisUrl
    $payload = @{ FunctionName = $FunctionName; Environment = @{ Variables = $LambdaVars } } | ConvertTo-Json -Depth 5
    $payloadFile = Join-Path $env:TEMP "arc-lambda-env-$PID.json"
    [System.IO.File]::WriteAllText($payloadFile, $payload, (New-Object System.Text.UTF8Encoding($false)))
    try {
        Invoke-Checked { aws lambda update-function-configuration --region $Region --cli-input-json ("file://" + ($payloadFile -replace "\\", "/")) | Out-Null }
    } finally {
        Remove-Item -LiteralPath $payloadFile -Force -ErrorAction SilentlyContinue
    }
    Invoke-Checked { aws lambda wait function-updated --function-name $FunctionName --region $Region }
}

Invoke-Checked { aws lambda update-function-code --function-name $FunctionName --image-uri $ImageSha --region $Region | Out-Null }
Invoke-Checked { aws lambda wait function-updated --function-name $FunctionName --region $Region }

if ($LambdaUrl) {
    try {
        $health = Invoke-RestMethod -Uri "$($LambdaUrl.TrimEnd('/'))/api/v1/health" -TimeoutSec 120
        Write-Host ("status={0} database={1} models={2} cache={3}" -f $health.status, $health.database_connected, $health.models_loaded, $health.cache_connected)
    } catch { Write-Warning "Health check failed: $_" }
}
Write-Host "==> Deployed $FunctionName ($GitSha)." -ForegroundColor Green
