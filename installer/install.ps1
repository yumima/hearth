# Install hearth, a local AI engine, and mantel, its chat window - for this user,
# without administrator rights (Windows PowerShell 5.1 or later).
#
#   irm https://yumima.github.io/hearth/install.ps1 | iex
#   & ([scriptblock]::Create((irm https://yumima.github.io/hearth/install.ps1))) --no-mantel
#
# Everything goes into one folder, and `hearth uninstall` removes it. An Ollama, a
# hearth or a Python that is already here is not touched. This script fetches uv
# and hearth, checks both against the hashes below, and lets hearth do the rest.
# (Plain ASCII: PowerShell 5.1 reads a file without a byte-order mark as ANSI.)
& {
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    $Version = '@VERSION@'
    $BaseUrl = '@BASE_URL@'
    $ManifestUrl = '@MANIFEST_URL@'
    $ManifestSha256 = '@MANIFEST_SHA256@'
    $Python = '@PYTHON@'
    $Wheel = '@WHEEL@'
    $WheelSha256 = '@WHEEL_SHA256@'
    $RequirementsSha256 = '@REQUIREMENTS_SHA256@'
    $UvVersion = '@UV_VERSION@'
    $UvFile = 'uv-x86_64-pc-windows-msvc.zip'
    $UvSha256 = '@UV_WINDOWS_X86_64@'

    function Get-Checked([string]$Url, [string]$File, [string]$Sha256) {
        if ((Test-Path -LiteralPath $File) -and ((Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash -eq $Sha256)) { return }
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile "$File.part"
        if ((Get-FileHash -LiteralPath "$File.part" -Algorithm SHA256).Hash -ne $Sha256) {
            Remove-Item -LiteralPath "$File.part"
            throw "$Url is not the published file"
        }
        Move-Item -Force -LiteralPath "$File.part" -Destination $File
    }
    # What a program writes to stderr is no error of ours; its exit code is.
    function Invoke-Step([string]$What, [string]$Program, [string[]]$Arguments) {
        $ErrorActionPreference = 'Continue'
        & $Program @Arguments
        if ($LASTEXITCODE -ne 0) { throw "$What failed ($LASTEXITCODE)" }
    }

    if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64' -and $env:PROCESSOR_ARCHITEW6432 -ne 'AMD64') {
        throw "there is no hearth for Windows on $env:PROCESSOR_ARCHITECTURE yet"
    }
    $Root = if ($env:HEARTH_DATA) { $env:HEARTH_DATA } else { Join-Path $env:LOCALAPPDATA 'hearth' }
    if ($Root -notmatch '^([A-Za-z]:\\|\\\\)') { throw 'HEARTH_DATA must be a full path' }
    if ((Test-Path -LiteralPath $Root) -and -not (Test-Path -LiteralPath (Join-Path $Root '.hearth-install'))) {
        $Others = @(Get-ChildItem -Force -LiteralPath $Root | Where-Object { $_.Name -notin @('models', 'voices') })
        if ($Others.Count) { throw "$Root is there already and was not made by this installer" }
    }

    Write-Output "Installing hearth $Version in $Root"
    $Down = Join-Path $Root "downloads\$Version"
    New-Item -ItemType Directory -Force -Path (Join-Path $Root 'bin'), $Down | Out-Null
    Set-Content -LiteralPath (Join-Path $Root '.hearth-install') -Value "hearth's installer made this folder, and removing hearth removes it."

    Get-Checked "https://github.com/astral-sh/uv/releases/download/$UvVersion/$UvFile" (Join-Path $Down $UvFile) $UvSha256
    Expand-Archive -Force -LiteralPath (Join-Path $Down $UvFile) -DestinationPath (Join-Path $Down 'uv')
    $Uv = Join-Path $Root 'bin\uv.exe'
    Copy-Item -Force -LiteralPath (Join-Path $Down 'uv\uv.exe') -Destination $Uv

    $WheelFile = Join-Path $Down $Wheel
    $Requirements = Join-Path $Down 'hearth-requirements.txt'
    Get-Checked "$BaseUrl$Wheel" $WheelFile $WheelSha256
    Get-Checked "${BaseUrl}hearth-requirements.txt" $Requirements $RequirementsSha256

    # Nothing of the person's own Python or uv settings has a part in it.
    $Saved = @{}
    $Ours = @{ UV_PYTHON_INSTALL_DIR = (Join-Path $Root 'python'); UV_CACHE_DIR = (Join-Path $Root 'cache'); UV_NO_CONFIG = '1'
               UV_PYTHON_PREFERENCE = 'only-managed'; UV_LINK_MODE = 'copy'; UV_DEFAULT_INDEX = 'https://pypi.org/simple'
               VIRTUAL_ENV = $null; PYTHONPATH = $null; PYTHONHOME = $null; PIP_INDEX_URL = $null; UV_INDEX_URL = $null
               UV_EXTRA_INDEX_URL = $null; UV_INDEX = $null }
    foreach ($Name in $Ours.Keys) {
        $Saved[$Name] = [Environment]::GetEnvironmentVariable($Name)
        [Environment]::SetEnvironmentVariable($Name, $Ours[$Name])
    }
    try {
        $Environment = Join-Path $Root "versions\$Version\hearth"
        $Complete = Join-Path $Environment '.complete'
        if (-not (Test-Path -LiteralPath $Complete)) {
            if (Test-Path -LiteralPath $Environment) { Remove-Item -Recurse -Force -LiteralPath $Environment }
            Write-Output 'Installing Python...'
            Invoke-Step 'Installing Python' $Uv @('python', 'install', '--quiet', $Python)
            Invoke-Step 'Making the environment' $Uv @('venv', '--quiet', '--python', $Python, $Environment)
            Write-Output 'Installing hearth...'
            Invoke-Step 'Installing what hearth needs' $Uv @('pip', 'install', '--quiet', '--python', $Environment, '--require-hashes', '--only-binary', ':all:', '-r', $Requirements)
            Invoke-Step 'Installing hearth' $Uv @('pip', 'install', '--quiet', '--python', $Environment, '--no-deps', '--no-index', $WheelFile)
            New-Item -ItemType File -Force -Path $Complete | Out-Null
        }
        try {
            Invoke-Step 'Installing hearth' (Join-Path $Environment 'Scripts\python.exe') (@('-I', '-m', 'hearth.install', 'finish', '--release', $ManifestUrl, '--sha256', $ManifestSha256) + $args)
        } catch {
            if ($LASTEXITCODE -eq 3 -and -not (Test-Path -LiteralPath (Join-Path $Root 'current.json'))) {
                # It stepped aside for something that is here: so does what was brought for it.
                foreach ($Part in 'bin', 'versions', 'python', 'ollama', 'cache', 'downloads', '.hearth-install') {
                    $Path = Join-Path $Root $Part
                    if (Test-Path -LiteralPath $Path) { Remove-Item -Recurse -Force -LiteralPath $Path -ErrorAction SilentlyContinue }
                }
                if (-not (Get-ChildItem -Force -LiteralPath $Root)) { Remove-Item -Force -LiteralPath $Root }
            }
            throw
        }
    } finally {
        foreach ($Name in $Saved.Keys) { [Environment]::SetEnvironmentVariable($Name, $Saved[$Name]) }
    }
} @args
