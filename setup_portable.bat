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
    echo.
    echo   [OK] Torch с CUDA уже работает -- пропускаем.
    goto :skip_torch
)

:: Сначала пробуем cu128 (Blackwell sm_90+)
echo.
echo   Установка Torch с CUDA...
echo   Попытка 1: cu128 (Blackwell / RTX 50xx)
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
if %errorlevel% neq 0 goto :try_cu126

python "%ROOT_DIR%app\check_torch.py"
if %errorlevel% equ 0 goto :skip_torch

:try_cu126
echo.
echo   Попытка 2: cu126 (Ampere / RTX 30-40xx)
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
if %errorlevel% neq 0 goto :try_cpu

python "%ROOT_DIR%app\check_torch.py"
if %errorlevel% equ 0 goto :skip_torch

:try_cpu
echo.
echo   [ПРЕДУПРЕЖДЕНИЕ] CUDA не встала, ставим CPU-версию...
uv pip install torch torchaudio

:skip_torch

:: ─── stable-audio-3 (без зависимостей, torch уже есть) ──
echo.
echo [3] Установка stable-audio-3...
uv pip install "git+https://github.com/Stability-AI/stable-audio-3.git" --no-deps
if %errorlevel% neq 0 (
    echo [ОШИБКА] Не удалось установить stable-audio-3
)

:: ─── Зависимости stable-audio-3 (кроме torch) ──────────
echo.
echo [3b] Установка зависимостей stable-audio-3...
uv pip install einops einops-exts huggingface-hub numpy safetensors soundfile tqdm transformers

:: ─── Веб-сервер ──────────────────────────────────────
echo.
echo [4] Установка веб-сервера и утилит...
uv pip install fastapi uvicorn requests python-multipart

:: ─── Проверка ────────────────────────────────────────
echo.
echo === Проверка установки ===
python "%ROOT_DIR%app\check_torch.py"
python -c "import stable_audio_3; print('  stable-audio-3: OK')" 2>nul || echo   [ПРЕДУПРЕЖДЕНИЕ] stable-audio-3
python -c "import fastapi; import uvicorn; import requests; print('  fastapi/uvicorn/requests: OK')"

echo.
echo ====================================================
echo   Установка завершена!
echo   Запустите run_portable.bat
echo ====================================================
pause
