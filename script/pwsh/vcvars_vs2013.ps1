# Find Visual Studio
# ECHO path to vcvarsall.bat or error string.
# Use [io.File]::Exists(...) to tell if found needed script.

$CurrentyDir = Split-Path -Parent $MyInvocation.MyCommand.Definition;

$vcvars = &"$CurrentyDir\vcvars_vs2013.bat"

if ([io.File]::Exists($vcvars)) {
    Write-Output $vcvars
    exit 0
}

Write-Error "$vcvars"

exit 1
