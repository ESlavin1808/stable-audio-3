@echo off
chcp 65001 >nul
title Stable Audio 3 — GUI

set "ROOT_DIR=%~dp0"
set "VENV_DIR=%ROOT_DIR%venv"

:: Перенаправляем кэш HuggingFace в папку программы
set "HF_HOME=%ROOT_DIR%hf_cache"
set "HUGGINGFACE_HUB_CACHE=%ROOT_DIR%hf_cache\hub"

if not exist "%VENV_DIR%" (
    echo [ОШИБКА] Виртуальное окружение не найдено.
    echo   Запустите сначала setup_portable.bat
    pause
    exit /b 1
)

if not exist "%ROOT_DIR%hf_cache" mkdir "%ROOT_DIR%hf_cache"
if not exist "%ROOT_DIR%hf_cache\hub" mkdir "%ROOT_DIR%hf_cache\hub"

echo ====================================================
echo   Stable Audio 3 — Графический интерфейс
echo ====================================================
echo.
"%VENV_DIR%\Scripts\python.exe" "%ROOT_DIR%app\detect_gpu.py"
echo.
echo   Все данные хранятся в папке программы.
echo   Для сброса удалите папки hf_cache/ и models/.
echo.
echo   Запуск: http://localhost:8765
echo   Для остановки закройте это окно
echo.

"%VENV_DIR%\Scripts\python.exe" "%ROOT_DIR%start_server.py"

echo.
pause
