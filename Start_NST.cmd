@echo off
setlocal
cd /d "%~dp0"
echo Starting the NST app. Keep this window open while using the app.
echo Once Flask says "Running on", open http://127.0.0.1:5000/ in your browser.
".venv\Scripts\python.exe" -m flask --app app run --host 127.0.0.1 --port 5000 --no-reload
if errorlevel 1 pause
