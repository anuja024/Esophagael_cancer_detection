@echo off
cd /d "%~dp0"

echo ================================
echo Launching Esophageal Cancer GUI
echo ================================

python cancer_gui.py

echo.
echo ================================
echo GUI closed. Press any key to exit.
echo ================================
pause
