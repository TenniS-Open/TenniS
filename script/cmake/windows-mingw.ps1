# Compile using Mingw

$CurrentyDir = Split-Path -Parent $MyInvocation.MyCommand.Definition;

$PROJECT = "$CurrentyDir\..\.."

cmake "$PROJECT" -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release $args


