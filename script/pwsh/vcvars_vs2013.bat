@echo off

set COMNTOOLS=%VS120COMNTOOLS%
set VCVARSALL="%COMNTOOLS%..\..\VC\vcvarsall.bat"

if "%COMNTOOLS%" == "" (
    goto :notfound
)

if not exist %VCVARSALL% (
	goto :filemissing
)

:succeed
echo %VCVARSALL%
exit 0


:notfound
echo Can not found VS2013: Not envriment named VS140COMNTOOLS.
exit 1


:filemissing
echo Can not found VS2013: File missing: %VCVARSALL%.
exit 2
