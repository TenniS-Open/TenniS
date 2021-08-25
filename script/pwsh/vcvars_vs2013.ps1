# Find shortcut

$CurrentyDir = Split-Path -Parent $MyInvocation.MyCommand.Definition;

$vcvars = &"$CurrentyDir\vcvars_vs2013.bat"

if ([io.File]::Exists($vcvars)) {
    Write-Output $vcvars
    exit 0
}

Write-Error "$vcvars"

exit 1
