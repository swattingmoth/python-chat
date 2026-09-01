param(
    [string]$ImageName = "python-chat-local",
    [string]$ContainerName = "python-chat-local",
    [int]$HostPort = 7860,
    [string]$EnvFile = ".env",
    [switch]$NoCache,
    [switch]$FollowLogs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Require-Command {
    param([string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found in PATH."
    }
}

Require-Command -Name "docker"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$envFilePath = Join-Path $projectRoot $EnvFile

if (-not (Test-Path $envFilePath)) {
    throw "Env file not found at '$envFilePath'. Create it or pass -EnvFile with a valid path."
}

$supabaseUrlLine = Get-Content $envFilePath | Where-Object { $_ -match '^SUPABASE_URL=' } | Select-Object -First 1
$supabaseUrl = $null
if ($supabaseUrlLine) {
    $supabaseUrl = ($supabaseUrlLine -replace '^SUPABASE_URL=', '').Trim()
}

$supabaseStorageContainerNameLine = Get-Content $envFilePath | Where-Object { $_ -match '^SUPABASE_STORAGE_CONTAINER_NAME=' } | Select-Object -First 1
$supabaseStorageContainerName = "supabase-storage"
if ($supabaseStorageContainerNameLine) {
    $parsedContainerName = ($supabaseStorageContainerNameLine -replace '^SUPABASE_STORAGE_CONTAINER_NAME=', '').Trim()
    if ($parsedContainerName) {
        $supabaseStorageContainerName = $parsedContainerName
    }
}

$imageBucketContainerPathLine = Get-Content $envFilePath | Where-Object { $_ -match '^SUPABASE_IMAGE_BUCKET_CONTAINER_PATH=' } | Select-Object -First 1
$supabaseImageBucketContainerPath = "/var/lib/storage/images"
if ($imageBucketContainerPathLine) {
    $parsedContainerPath = ($imageBucketContainerPathLine -replace '^SUPABASE_IMAGE_BUCKET_CONTAINER_PATH=', '').Trim()
    if ($parsedContainerPath) {
        $supabaseImageBucketContainerPath = $parsedContainerPath
    }
}

$supabaseStorageContainerId = ""
$supabaseStorageContainerOutput = & docker ps -aq --filter "name=^${supabaseStorageContainerName}$"
if ($null -ne $supabaseStorageContainerOutput) {
    $supabaseStorageContainerId = ($supabaseStorageContainerOutput | Select-Object -First 1).ToString().Trim()
}

$containerSupabaseUrl = $supabaseUrl
if ($supabaseUrl) {
    $containerSupabaseUrl = $supabaseUrl -replace '://127\.0\.0\.1', '://host.docker.internal'
    $containerSupabaseUrl = $containerSupabaseUrl -replace '://localhost', '://host.docker.internal'
}

if ($supabaseUrl -and ($supabaseUrl -match 'localhost|127\.0\.0\.1')) {
    Write-Host "Warning: SUPABASE_URL uses localhost/127.0.0.1. Inside Docker this points to the container itself." -ForegroundColor Yellow
    Write-Host "Hint: use host.docker.internal in SUPABASE_URL for host services (example: http://host.docker.internal:54321)." -ForegroundColor Yellow
    if ($containerSupabaseUrl) {
        Write-Host "Applying container override: SUPABASE_URL=$containerSupabaseUrl" -ForegroundColor Yellow
    }
}

if ($supabaseStorageContainerId) {
    Write-Host "Sharing volumes from Supabase container '$supabaseStorageContainerName' (read-only)." -ForegroundColor Yellow
    Write-Host "Using image bucket container path: $supabaseImageBucketContainerPath" -ForegroundColor Yellow
} else {
    Write-Host "Supabase storage container '$supabaseStorageContainerName' not found. Image URL mapping to local file paths is disabled." -ForegroundColor Yellow
    Write-Host "Set SUPABASE_STORAGE_CONTAINER_NAME in $EnvFile if your container name differs." -ForegroundColor Yellow
}

Push-Location $projectRoot
try {
    $buildArgs = @("build", "-t", $ImageName)
    if ($NoCache) {
        $buildArgs += "--no-cache"
    }
    $buildArgs += "."

    Write-Host "Building image '$ImageName'..." -ForegroundColor Cyan
    & docker @buildArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Docker build failed."
    }

    $existingContainerOutput = & docker ps -aq --filter "name=^${ContainerName}$"
    $existingContainerId = ""
    if ($null -ne $existingContainerOutput) {
        $existingContainerId = ($existingContainerOutput | Select-Object -First 1).ToString().Trim()
    }

    if ($existingContainerId) {
        Write-Host "Removing existing container '$ContainerName'..." -ForegroundColor Yellow
        & docker rm -f $ContainerName | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to remove existing container '$ContainerName'."
        }
    }

    Write-Host "Starting container '$ContainerName' on http://localhost:$HostPort ..." -ForegroundColor Cyan
    $runArgs = @(
        "run",
        "-d",
        "--name",
        $ContainerName,
        "--env-file",
        $envFilePath,
        "--add-host",
        "host.docker.internal:host-gateway",
        "-e",
        "PORT=7860"
    )
    if ($containerSupabaseUrl) {
        $runArgs += @("-e", "SUPABASE_URL=$containerSupabaseUrl")
    }
    if ($supabaseStorageContainerId) {
        $runArgs += @("--volumes-from", "${supabaseStorageContainerId}:ro")
        $runArgs += @("-e", "SUPABASE_PUBLIC_IMAGE_BUCKET_MOUNT_PATH=$supabaseImageBucketContainerPath")
    }
    $runArgs += @("-p", "${HostPort}:7860", $ImageName)

    & docker @runArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Docker run failed."
    }

    $healthUrl = "http://localhost:$HostPort/healthz"
    $readyDeadline = (Get-Date).AddSeconds(30)
    $isHealthy = $false

    while ((Get-Date) -lt $readyDeadline) {
        $runningRaw = & docker inspect -f "{{.State.Running}}" $ContainerName 2>$null
        if ($LASTEXITCODE -ne 0 -or $null -eq $runningRaw -or $runningRaw.Trim() -ne "true") {
            Write-Host "Container failed to stay running. Recent logs:" -ForegroundColor Red
            $failureLogs = & docker logs --tail 120 $ContainerName
            $failureLogs
            if ($failureLogs -match 'ConnectError|Connection refused') {
                Write-Host "Likely cause: container cannot reach Supabase endpoint." -ForegroundColor Yellow
                Write-Host "Check SUPABASE_URL in $EnvFile and prefer host.docker.internal over localhost when targeting host services." -ForegroundColor Yellow
            }
            throw "Container '$ContainerName' exited during startup."
        }

        try {
            $healthResponse = Invoke-WebRequest -Uri $healthUrl -Method Get -UseBasicParsing -TimeoutSec 2
            if ($healthResponse.StatusCode -eq 200) {
                $isHealthy = $true
                break
            }
        }
        catch {
            # Continue retrying until timeout to allow API startup time.
        }

        Start-Sleep -Seconds 1
    }

    if (-not $isHealthy) {
        Write-Host "Container did not become healthy within startup timeout. Recent logs:" -ForegroundColor Red
        $timeoutLogs = & docker logs --tail 120 $ContainerName
        $timeoutLogs
        if ($timeoutLogs -match 'ConnectError|Connection refused') {
            Write-Host "Likely cause: container cannot reach Supabase endpoint." -ForegroundColor Yellow
            Write-Host "Check SUPABASE_URL in $EnvFile and prefer host.docker.internal over localhost when targeting host services." -ForegroundColor Yellow
        }
        throw "Container '$ContainerName' did not report a healthy /healthz response."
    }

    Write-Host "Container started successfully." -ForegroundColor Green
    Write-Host "App URL:      http://localhost:$HostPort/app" -ForegroundColor Green
    Write-Host "Health URL:   http://localhost:$HostPort/healthz" -ForegroundColor Green
    Write-Host "Readiness URL:http://localhost:$HostPort/readyz" -ForegroundColor Green

    if ($FollowLogs) {
        Write-Host "Tailing logs for '$ContainerName' (Ctrl+C to stop)..." -ForegroundColor Cyan
        & docker logs -f $ContainerName
    }
}
finally {
    Pop-Location
}
