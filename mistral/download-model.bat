@echo off
start /b ollama serve
timeout /t 5 /nobreak
ollama pull mistral
taskkill /f /im ollama.exe
xcopy /E /I /Y "%USERPROFILE%\.ollama\models" models\ 