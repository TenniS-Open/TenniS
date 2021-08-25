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

