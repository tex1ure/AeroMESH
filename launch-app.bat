@echo off
title AeroMESH Desktop Studio
cd /d "%~dp0"
echo ========================================================================
echo    Launching AeroMESH Unified Desktop Studio (CUDA + Mesh)
echo ========================================================================

if exist "target\release\AeroMesh-Desktop.exe" (
    start "" "target\release\AeroMesh-Desktop.exe"
) else if exist "bin\AeroMesh-Desktop.exe" (
    start "" "bin\AeroMesh-Desktop.exe"
) else if exist "AeroMesh-Desktop.exe" (
    start "" "AeroMesh-Desktop.exe"
) else if exist "target\debug\AeroMesh-Desktop.exe" (
    start "" "target\debug\AeroMesh-Desktop.exe"
) else (
    echo Building AeroMesh-Desktop.exe...
    cargo build --bin AeroMesh-Desktop
    if exist "target\debug\AeroMesh-Desktop.exe" (
        start "" "target\debug\AeroMesh-Desktop.exe"
    ) else (
        echo [!] Failed to find or compile AeroMesh-Desktop.exe.
        pause
    )
)
