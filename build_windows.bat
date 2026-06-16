@echo off
setlocal

python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt
python -m PyInstaller --clean motion-planning.spec

echo.
echo Build complete.
echo EXE path: dist\motion-planning.exe
pause
