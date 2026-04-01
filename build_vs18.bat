@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 (
    echo ERROR: vcvarsall failed
    exit /b 1
)
cd /d "%~dp0"
cmake --preset %1
if errorlevel 1 exit /b 1
cmake --build --preset %1
