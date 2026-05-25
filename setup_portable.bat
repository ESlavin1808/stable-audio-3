@echo off
chcp 65001 >nul
title Stable Audio 3 — Установка

echo ====================================================
echo   Stable Audio 3 — Портативная версия
echo   Установка зависимостей
echo ====================================================
echo.
echo   * Это установка только кода и библиотек
echo   * Модели НЕ скачиваются — это делается из GUI
echo   * После установки запустите run_portable.bat
echo.

set "ROOT_DIR=%~dp0"
set "VENV_DIR=%ROOT_DIR%venv"
set "OUTPUT_DIR=%ROOT_DIR%output"
set "HF_CACHE_DIR=%ROOT_DIR%hf_cache"

:: Проверка uv
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [ОШИБКА] uv не найден. Установите uv:
    echo   pip install uv
    echo   или:  https://docs.astral.sh/uv/
    pause
    exit /b 1
)

:: Создаём вспомогательные папки
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%HF_CACHE_DIR%" mkdir "%HF_CACHE_DIR%"
if not exist "%HF_CACHE_DIR%\hub" mkdir "%HF_CACHE_DIR%\hub"

echo [1/4] Создание виртуального окружения Python...
if exist "%VENV_DIR%" (
    echo   Виртуальное окружение уже есть, очищаем...
    rmdir /s /q "%VENV_DIR%"
)
uv venv "%VENV_DIR%" --python 3.10
if %errorlevel% neq 0 (
    uv venv "%VENV_DIR%"
)

echo [2/4] Установка PyTorch с CUDA 12.6...
call "%VENV_DIR%\Scripts\activate.bat"
pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
if %errorlevel% neq 0 (
    echo [ПРЕДУПРЕЖДЕНИЕ] PyTorch CUDA не установился, пробуем CPU...
    pip install torch torchaudio --force-reinstall
)

echo [3/4] Установка stable-audio-3 из GitHub...
pip install git+https://github.com/Stability-AI/stable-audio-3.git

echo [4/4] Установка веб-сервера...
pip install fastapi uvicorn

:: Проверка
echo.
echo === Проверка ===
python -c "import stable_audio_3; print('  stable-audio-3: OK')" 2>nul || echo "  [ПРЕДУПРЕЖДЕНИЕ] stable-audio-3 не импортируется"
python -c "import torch; print(f'  Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import requests; print('  requests: OK')"
python -c "import soundfile; print('  soundfile: OK')"

echo.
echo ====================================================
echo   Установка завершена!
echo.
echo   Модели НЕ скачивались. Запустите run_portable.bat
echo   и управляйте моделями из интерфейса.
echo ====================================================
pause
