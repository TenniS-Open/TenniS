@echo off
:: Find Visual Studio
:: ECHO path to vcvarsall.bat or error string.
:: Use `if exist ... (...)` to tell if found needed script.

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
echo Can not found VS2013: Envriment variable missing: %%VS120COMNTOOLS%%.
exit 1


:filemissing
echo Can not found VS2013: File missing: %VCVARSALL%.
exit 2
