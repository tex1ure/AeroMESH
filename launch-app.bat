@echo off
title AeroMESH Desktop App
cd /d "%~dp0"
echo Launching AeroMESH Desktop Window...
if exist "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" (
    start "" "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --app="http://127.0.0.1:7860" --window-size=1300,860
) else if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --app="http://127.0.0.1:7860" --window-size=1300,860
) else (
    start http://127.0.0.1:7860
)
