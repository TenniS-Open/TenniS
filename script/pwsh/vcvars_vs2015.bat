@echo off
:: Find Visual Studio
:: ECHO path to vcvarsall.bat or error string.
:: Use `if exist ... (...)` to tell if found needed script.

set COMNTOOLS=%VS140COMNTOOLS%
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
echo Can not found VS2015: Envriment variable missing: %%VS140COMNTOOLS%%.
exit 1


:filemissing
echo Can not found VS2015: File missing: %VCVARSALL%.
exit 2
