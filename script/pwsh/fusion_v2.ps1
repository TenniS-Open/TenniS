# Fusion dll (or exe) references to present directory.
# Usage: <command> target
# [target]: path to target dll or exe. If target is directory, all DLLs and EXEs will be fusioned by default.
# Notice: The search order is: %PATH%, C:\Windows\System32, C:\Windows\SysWOW64

param(
    [string]$target
)

$CurrentyDir = Split-Path -Parent $MyInvocation.MyCommand.Definition;

$tasks = @()

if ([io.File]::Exists($target)) {
    # Write-Output "Package file: $target"
    $path = Resolve-Path $target
    $tasks += $path.Path
} elseif ([io.Directory]::Exists($target)) {
    # Write-Output "Package directory: $target"
    foreach ($_ in Get-ChildItem "$target" -Filter *.exe -Recurse) {
        $tasks += $_.FullName
    }
    foreach ($_ in Get-ChildItem "$target" -Filter *.dll -Recurse) {
        $tasks += $_.FullName
    }
} else {
    Write-Error "Can not access: $target"
    exit 1
}

foreach($task in $tasks) {
    &"$CurrentyDir\fusion.ps1" "$task"
}

