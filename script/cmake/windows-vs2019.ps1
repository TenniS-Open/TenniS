# Compile using MSVC

# ARCH: x86 x64
# CUDA: ON OFF

$COMPILER = "vs2019"
$ARCH = "x64"
$CUDA = "OFF"

$CurrentyDir = Split-Path -Parent $MyInvocation.MyCommand.Definition;

$PROJECT = "$CurrentyDir\..\.."

&"$CurrentyDir\..\pwsh\setup_vcvars.ps1" "$COMPILER" "$ARCH"

cmake "$PROJECT" -G "NMake Makefiles" -DTS_DYNAMIC_INSTRUCTION=ON -DTS_USE_CUDA="$CUDA" -DCMAKE_BUILD_TYPE=Release $args


