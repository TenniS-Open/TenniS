# Find Visual Studio
# ECHO path to vcvarsall.bat or error string.
# Use [io.File]::Exists(...) to tell if found needed script.

$startup = "Visual Studio 2015"
$ide2buildtools = "..\..\VC"
$buildtools2vcvarsall = "vcvarsall.bat"

$shell = New-Object -ComObject WScript.Shell

$pathUser = [System.Environment]::GetFolderPath('StartMenu')
$pathCommon = $shell.SpecialFolders.Item('AllUsersStartMenu')

$alternative = @(
    [io.path]::combine($pathUser, $startup)
    [io.path]::combine($pathUser, "Programs", $startup)
    [io.path]::combine($pathCommon, $startup)
    [io.path]::combine($pathCommon, "Programs", $startup)
    )

$foundShortcut = 1

foreach($a in $alternative) {
    if ([io.File]::Exists($a)) {
        $foundShortcut = 0
        $link = $shell.CreateShortcut($a.FullName)
        $vavars = [io.path]::combine($link.WorkingDirectory, $ide2buildtools, $buildtools2vcvarsall)
        if ([io.File]::Exists($vavars)) {
            Write-Output $vavars
            exit 0
        }
    } elseif ([io.Directory]::Exists($a)) {
        # Write-Output Checking *.lnk in $a
        $shortcuts = Get-ChildItem "$a" -Filter *.lnk  -Recurse 
        foreach ($b in $shortcuts) {
            $foundShortcut = 0
            $link = $shell.CreateShortcut($b.FullName)
            $vavars = [io.path]::combine($link.WorkingDirectory, $buildtools2vcvarsall)
            if ([io.File]::Exists($vavars)) {
                Write-Output $vavars
                exit 0
            }
        }
    }
}

if ($foundShortcut) {
    Write-Error "Can not found ${startup}: No shortcut in StartMenu."
} else {
    Write-Error "Can not found ${startup}: File missing: $buildtools2vcvarsall."
}

exit 1