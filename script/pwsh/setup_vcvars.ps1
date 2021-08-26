# Setup VC envriment
# Usage: <command> version [arch]
# [version]: vs2013 | vs2015 | vs2017 | vs2019
# [arch]: x86 | {amd64} | x86_amd64 | x86_arm | x86_arm64 | amd64_x86 | amd64_arm | amd64_arm64
# Notice: before use this script, please install according Visual Studio and create Shotcut at Start Menu by default.

param (
    [string]$version,
    [string]$arch = "amd64"
 )

$toolset = @{
    vs2013 = "vcvars_vs2013.ps1";
    vs2015 = "vcvars_vs2015.ps1";
    vs2017 = "vcvars_vs2017.ps1";
    vs2019 = "vcvars_vs2019.ps1"
}

$CurrentyDir = Split-Path -Parent $MyInvocation.MyCommand.Definition;

if ($null -eq $toolset[$version]) {
    $keys = [string]$toolset.Keys
    Write-Output "Usage: <command> version [arch]"
    Write-Output "[version]: $keys"
    Write-Output "[arch]: x86 | {amd64} | x86_amd64 | x86_arm | x86_arm64 | amd64_x86 | amd64_arm | amd64_arm64"
    Write-Error "version must be: $keys"
    exit 1
}

$get_vcvars_bat = [io.path]::combine($CurrentyDir, $toolset[$version])

$vcvars_bat = &"$get_vcvars_bat"

if (![io.File]::Exists($vcvars_bat)) {
    Write-Error $vcvars_bat
    exit 2
}

$command = "`"$vcvars_bat`" $arch"

foreach($_ in cmd /c ($command + " > NUL 2>&1 && SET")) {
    if ($_ -match '^([^=]+)=(.*)') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}
