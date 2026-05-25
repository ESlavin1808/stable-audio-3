@echo off
chcp 65001 >nul
title Stable Audio 3 — Установка

echo ====================================================
echo   Stable Audio 3 — Портативная версия
echo ====================================================
echo.

set "ROOT_DIR=%~dp0"
set "VENV_DIR=%ROOT_DIR%venv"

:: uv обязателен
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [ОШИБКА] uv не найден.
    echo   Установите:  pip install uv
    pause
    exit /b 1
)

:: ─── Виртуальное окружение ─────────────────────────────
if exist "%VENV_DIR%" (
    echo [1] Виртуальное окружение уже есть (%VENV_DIR%^)
) else (
    echo [1] Создание виртуального окружения...
    uv venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo [ОШИБКА] Не удалось создать виртуальное окружение
        pause
        exit /b 1
    )
)

call "%VENV_DIR%\Scripts\activate.bat"

:: ─── Проверка torch + CUDA ────────────────────────────
echo.
echo [2] Проверка существующего Torch...
python "%ROOT_DIR%app\check_torch.py"
set "TORCH_STATUS=%errorlevel%"

if %TORCH_STATUS% equ 0 (
    echo   Torch с CUDA уже работает — пропускаем переустановку.
    goto :skip_torch
)
if %TORCH_STATUS% equ 1 (
    echo.
    echo   Torch есть но без CUDA. Переустанавливаем...
)
if %TORCH_STATUS% equ 2 (
    echo.
    echo   Устанавливаем Torch с поддержкой CUDA...
)

:: Установка torch с CUDA
echo.
echo   Установка: uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
if %errorlevel% neq 0 (
    echo   [ПРЕДУПРЕЖДЕНИЕ] CUDA-версия не встала, пробуем CPU...
    uv pip install torch torchaudio
)
:skip_torch

:: ─── stable-audio-3 ───────────────────────────────────
echo.
echo [3] Установка stable-audio-3...
uv pip install git+https://github.com/Stability-AI/stable-audio-3.git
if %errorlevel% neq 0 (
    echo [ОШИБКА] Не удалось установить stable-audio-3
)

:: ─── Веб-сервер ──────────────────────────────────────
echo.
echo [4] Установка веб-сервера и утилит...
uv pip install fastapi uvicorn requests soundfile python-multipart

:: ─── Проверка ────────────────────────────────────────
echo.
echo === Проверка установки ===
python -c "import torch; print(f'  Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
if %errorlevel% neq 0 echo   [ОШИБКА] Torch не импортируется
python -c "import stable_audio_3; print('  stable-audio-3: OK')" 2>nul || echo   [ПРЕДУПРЕЖДЕНИЕ] stable-audio-3 не импортируется
python -c "import fastapi; import uvicorn; import requests; print('  fastapi/uvicorn/requests: OK')"

echo.
echo ====================================================
echo   Установка завершена!
echo.
echo   Запустите run_portable.bat
echo ====================================================
pause
