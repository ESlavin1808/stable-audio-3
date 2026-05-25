@echo off
chcp 65001 >nul
title Stable Audio 3 — Установка

echo ====================================================
echo   Stable Audio 3 — Портативная версия
echo   Автоматическая установка зависимостей
echo ====================================================
echo.
echo   * Определяем GPU и CUDA — подбираем правильный PyTorch
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
    echo [ОШИБКА] uv не найден.
    echo   Установите uv:   pip install uv
    echo   Или скачайте:    https://docs.astral.sh/uv/#installation
    pause
    exit /b 1
)

:: Создаём вспомогательные папки
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%HF_CACHE_DIR%" mkdir "%HF_CACHE_DIR%"
if not exist "%HF_CACHE_DIR%\hub" mkdir "%HF_CACHE_DIR%\hub"

:: ─── Шаг 1: Python 3.10 — виртуальное окружение ────────────────────
echo.
echo [1/4] Создание виртуального окружения Python 3.10...
if exist "%VENV_DIR%" (
    echo   Виртуальное окружение уже есть, очищаем...
    rmdir /s /q "%VENV_DIR%"
)
uv venv "%VENV_DIR%" --python 3.10
if %errorlevel% neq 0 (
    echo   Python 3.10 не найден, пробуем 3.11...
    uv venv "%VENV_DIR%" --python 3.11
)
if %errorlevel% neq 0 (
    echo   Python 3.11 не найден, пробуем системный Python 3...
    uv venv "%VENV_DIR%"
)
echo   Виртуальное окружение создано.

call "%VENV_DIR%\Scripts\activate.bat"

:: ─── Шаг 2: PyTorch — автоопределение GPU/CUDA ────────────────────
echo.
echo [2/4] Определение GPU и установка PyTorch...
echo.

python "%ROOT_DIR%app\detect_gpu.py"

echo.
python "%ROOT_DIR%app\detect_gpu.py" --install "%VENV_DIR%\Scripts\python.exe"
if %errorlevel% neq 0 (
    echo [ПРЕДУПРЕЖДЕНИЕ] PyTorch не установился стандартным способом.
    echo   Пробуем pip install напрямую...
    pip install torch torchaudio --force-reinstall
)

:: ─── Шаг 3: stable-audio-3 из GitHub ──────────────────────────
echo.
echo [3/4] Установка stable-audio-3 из GitHub...
pip install git+https://github.com/Stability-AI/stable-audio-3.git
if %errorlevel% neq 0 (
    echo [ОШИБКА] Не удалось установить stable-audio-3
    echo   Проверьте соединение с GitHub
)

:: ─── Шаг 4: Веб-сервер и прочие зависимости ────────────────────
echo.
echo [4/4] Установка веб-сервера и утилит...
pip install fastapi uvicorn requests

:: ─── Проверка ─────────────────────────────────────────
echo.
echo === Проверка установки ===
python -c "import torch; print(f'  Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import stable_audio_3; print('  stable-audio-3: OK')" 2>nul || echo "  [ПРЕДУПРЕЖДЕНИЕ] stable-audio-3 не импортируется"
python -c "import fastapi; import uvicorn; import requests; import soundfile; print('  fastapi/uvicorn/requests/soundfile: OK')"

echo.
echo ====================================================
echo   Установка завершена!
echo.
echo   Модели НЕ скачивались. Запустите run_portable.bat
echo   и управляйте моделями из интерфейса.
echo ====================================================
pause
