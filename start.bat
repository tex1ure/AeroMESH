@echo off
title AeroMesh Cluster Launcher
cd /d "%~dp0"
powershell.exe -ExecutionPolicy Bypass -NoProfile -File "%~dp0start.ps1" %*
