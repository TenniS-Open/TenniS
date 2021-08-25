param(
    [string]$target
)

$ignore = @{}
$ignore["kernel32.dll"] = 1

if ([io.File]::Exists($target)) {
    $target = (Resolve-Path $target).Path
} else {
    Write-Error "Can not access: $target"
    exit 1
}

$root = Split-Path -Parent $target
$file = Split-Path -Leaf $target

$ready_dlls = @{}
$walked_dlls = @{}

foreach ($_ in Get-ChildItem "$root" -Filter *.dll -Recurse) {
    $ref = Split-Path -Leaf $_.FullName
    $ref = $ref.trim().ToLower()
    $ready_dlls[$ref] = 1
}

function Get-Arch {
    param(
        [string]$dll
    )
    $arch = dumpbin.exe /headers $dll | where {$_ -match "machine \((.*)\)"}
    return $matches[1]
}

function Fusion {
    param(
        [string]$dll,
        [string]$arch
    )
    $refs = dumpbin.exe /dependents $dll | where {$_ -match "\.dll$"}
    :refs
    foreach ($ref in $refs) {
        # ignore self
        if ($ref -match " " + ($dll -replace "\.", "\.") + "$") {
            continue refs
        }
        $ref = $ref.trim()
        $lower_ref = $ref.ToLower()
        # the ref had walked
        if ($null -ne $walked_dlls[$lower_ref]) {
            continue refs
        }
        $walked_dlls[$lower_ref] = 1    # now i had walked this dll
        # ignore system DLL
        if ($null -ne $ignore[$lower_ref]) {
            continue refs
        }
        # ignore windows API
        if ($lower_ref -match "^ext-ms-") {
            continue refs
        }
        if ($lower_ref -match "^api-ms-") {
            continue refs
        }
        # check if dll exists
        if ($null -eq $ready_dlls[$lower_ref]) {
            # find and copy dll
            ## find
            $posible = @()
            foreach ($_ in where.exe $ref) { $posible += $_; }
            ### try x86 and x64 as well
            foreach ($_ in where.exe $ref /R C:\Windows\System32) { $posible += $_; }
            foreach ($_ in where.exe $ref /R C:\Windows\SysWOW64) { $posible += $_; }
            $found = $null
            :posible
            foreach ($_ in $posible) {
                $a = Get-Arch $_
                if ($null -eq $a) {
                    continue posible
                }
                if ($arch -ne $a) {
                    continue posible
                }
                $found = $_
                break
            }
            ## check
            if ($null -eq $found) {
                Write-Output "[WARNING] Can not found $ref for $dll($arch)."
                continue refs
            }
            ## copy
            Write-Output "[INFO] Fusion $dll <- $found"
            Copy-Item $found .
            $ready_dlls[$lower_ref] = 1
            $ref = Split-Path -Leaf $found  # make ref name to found ref.
        }
        # now walk next ref
        Fusion $ref $arch
    }
}

$arch = Get-Arch $target

$pwd = (pwd).Path

cd $root
Fusion $file $arch
cd $pwd