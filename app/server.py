"""
Stable Audio 3 — Portable GUI Backend
FastAPI сервер для генерации аудио через Stable Audio 3.
"""

import os
import time
import uuid
import asyncio
import logging
import tempfile
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stable-audio-3")

# Пути
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
STATIC_DIR = ROOT_DIR / "app" / "static"
HF_CACHE_DIR = ROOT_DIR / "hf_cache"
MODELS_DIR = ROOT_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
HF_CACHE_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Перенаправляем кэш HuggingFace в папку программы (портабельность)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
# Создаём структуру HF-кэша в папке программы
(HF_CACHE_DIR / "hub").mkdir(exist_ok=True)

# Состояние модели (ленивая загрузка)
_model_instance = {"model": None, "model_name": None, "device": None}

# Настройки
SETTINGS_FILE = ROOT_DIR / "settings.json"
_hf_token = None
_offline_mode = False
_local_models_dir = ""  # путь к папке с локальными моделями
_custom_output_dir = ""  # пользовательская папка для выходных файлов

# Список моделей (repo_id, config_file, ckpt_file)
AVAILABLE_MODELS = [
    {"id": "small-music", "name": "Stable Audio 3 Small (Music)", "params": "433M", "device": "CPU/GPU", "max_duration": 120, "description": "Музыка, до 120 с", "repo": "stabilityai/stable-audio-3-small-music"},
    {"id": "small-sfx",   "name": "Stable Audio 3 Small (SFX)",   "params": "433M", "device": "CPU/GPU", "max_duration": 120, "description": "Звуковые эффекты, до 120 с", "repo": "stabilityai/stable-audio-3-small-sfx"},
    {"id": "medium",      "name": "Stable Audio 3 Medium",         "params": "1.4B", "device": "GPU",    "max_duration": 380, "description": "Высокое качество, до 380 с (GPU)", "repo": "stabilityai/stable-audio-3-medium"},
]

# HuggingFace license URLs
LICENSE_URLS = {
    "small-music": "https://huggingface.co/stabilityai/stable-audio-3-small-music",
    "small-sfx": "https://huggingface.co/stabilityai/stable-audio-3-small-sfx",
    "medium": "https://huggingface.co/stabilityai/stable-audio-3-medium",
}

def _load_settings():
    global _hf_token, _offline_mode, _local_models_dir, _custom_output_dir, OUTPUT_DIR
    if SETTINGS_FILE.exists():
        try:
            import json
            with open(SETTINGS_FILE) as f:
                data = json.load(f)
            _hf_token = data.get("hf_token", None)
            _offline_mode = data.get("offline", False)
            _local_models_dir = data.get("local_models_dir", "")
            _custom_output_dir = data.get("output_dir", "")
            if _custom_output_dir:
                custom = Path(_custom_output_dir)
                custom.mkdir(parents=True, exist_ok=True)
                OUTPUT_DIR = custom
            # Восстанавливаем env var для HF (офлайн-режим)
            if _offline_mode:
                os.environ["HF_HUB_OFFLINE"] = "1"
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)
        except Exception:
            pass

def _save_settings():
    import json
    data = {
        "offline": _offline_mode,
        "local_models_dir": _local_models_dir,
        "output_dir": _custom_output_dir,
    }
    if _hf_token:
        data["hf_token"] = _hf_token
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)

_load_settings()

app = FastAPI(title="Stable Audio 3", version="1.0.0")

# CORS для локального доступа
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Вспомогательные функции ──────────────────────────────────────────────

def _load_audio_from_bytes(file_bytes: bytes, file_ext: str) -> tuple:
    """
    Загружает аудио из байтов в numpy array.
    Возвращает (sample_rate, audio_tensor) для StableAudioModel.
    """
    import soundfile as sf
    import torch

    # Сохраняем во временный файл для soundfile
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # Читаем через soundfile
        data, sr = sf.read(tmp_path)
        # Конвертируем в torch tensor [channels, samples]
        if data.ndim == 1:
            data = np.expand_dims(data, 0)  # [1, samples]
        else:
            data = data.T  # [channels, samples]
        audio_tensor = torch.from_numpy(data).float()
        return (sr, audio_tensor)
    finally:
        os.unlink(tmp_path)


# ─── Менеджер модели ──────────────────────────────────────────────────────

def _find_local_model(model_id: str) -> Optional[tuple[str, str]]:
    """
    Ищет модель в локальной папке.
    Возвращает (config_path, ckpt_path) или None.
    """
    if not _local_models_dir:
        return None
    model_dir = Path(_local_models_dir) / model_id
    if not model_dir.exists():
        return None
    config_path = model_dir / "model_config.json"
    ckpt_files = list(model_dir.glob("*.safetensors"))
    if config_path.exists() and ckpt_files:
        return str(config_path), str(ckpt_files[0])
    return None


def _load_model_from_local(model_id: str, config_path: str, ckpt_path: str, device: str):
    """Загружает модель напрямую из локальных файлов, минуя HuggingFace."""
    import json
    import torch
    from stable_audio_3 import StableAudioModel
    from stable_audio_3.loading_utils import load_diffusion_cond

    log.info(f"Загрузка из локальных файлов: {ckpt_path}")
    with open(config_path) as f:
        model_config = json.load(f)

    model_half = device == "cuda"
    loaded = load_diffusion_cond(model_config, ckpt_path, device=device, model_half=model_half)
    return StableAudioModel(loaded, model_config, device, model_half)


def get_model(model_name: str = "small-music"):
    """
    Загружает модель StableAudioModel.
    Приоритет: 1) локальные файлы  2) HF Hub (с токеном)
    """
    global _model_instance

    if _model_instance["model"] is not None and _model_instance["model_name"] == model_name:
        return _model_instance["model"]

    log.info(f"Загрузка модели: {model_name}...")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Пробуем локальные файлы
    local = _find_local_model(model_name)
    if local:
        model = _load_model_from_local(model_name, local[0], local[1], device)
        _model_instance = {"model": model, "model_name": model_name, "device": device}
        log.info(f"Модель {model_name} загружена из локальных файлов на {device}")
        return model

    # 2. Загрузка с HuggingFace Hub
    try:
        from stable_audio_3 import StableAudioModel
        from huggingface_hub import login

        if _offline_mode:
            os.environ["HF_HUB_OFFLINE"] = "1"
            log.info("Офлайн-режим: используем только локальный кэш HF")
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
            if _hf_token:
                login(token=_hf_token, add_to_git_credential=False)
                log.info("HuggingFace: аутентифицирован по токену")

        model = StableAudioModel.from_pretrained(model_name, model_half=device == "cuda")
        _model_instance = {"model": model, "model_name": model_name, "device": device}
        log.info(f"Модель {model_name} загружена с HuggingFace на {device}")
        return model

    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Ошибка импорта stable_audio_3: {e}. Установите через setup_portable.bat")
    except Exception as e:
        err = str(e)
        if "403" in err or "gated" in err or "access" in err.lower():
            raise HTTPException(status_code=403, detail=f"Нет доступа к модели {model_name}. Примите лицензию на сайте HuggingFace и укажите токен в настройках.")
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки модели {model_name}: {e}")


# ─── API endpoints ───────────────────────────────────────────────────────

@app.get("/api/models")
async def list_models():
    """Список доступных моделей с указанием статуса доступности."""
    result = []
    for m in AVAILABLE_MODELS:
        status = "download"  # нужно скачать
        source = None

        # Проверяем локальные файлы
        local = _find_local_model(m["id"])
        if local:
            status = "local"
            source = local[0]

        # Проверяем HF кэш
        if status != "local":
            from huggingface_hub import try_to_load_from_cache
            cached_config = try_to_load_from_cache(m["repo"], "model_config.json")
            cached_ckpt = try_to_load_from_cache(m["repo"], "model.safetensors")
            if isinstance(cached_config, str) and isinstance(cached_ckpt, str):
                status = "cached"
                source = "HF cache"

        entry = dict(m)
        entry["status"] = status
        entry["source"] = source
        entry["license_url"] = LICENSE_URLS.get(m["id"], "")
        result.append(entry)

    return {"models": result, "local_models_dir": _local_models_dir, "offline": _offline_mode, "token_configured": bool(_hf_token)}


@app.get("/api/device")
async def get_device():
    """Информация о доступном устройстве (CPU/GPU) и настройках."""
    try:
        import torch
        info = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "torch_version": torch.__version__,
            "hf_token_configured": _hf_token is not None and len(_hf_token) > 0,
            "offline_mode": _offline_mode,
            "local_models_dir": _local_models_dir,
        }
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
            info["vram_total"] = f"{vram / 1024**3:.1f} GB"
        return info
    except Exception as e:
        return {"device": "unknown", "error": str(e)}


# ─── Управление моделями ─────────────────────────────────────────────────

@app.post("/api/models/set-path")
async def set_models_path(data: dict = Body(...)):
    """Устанавливает путь к папке с локальными моделями."""
    global _local_models_dir
    path = data.get("path", "").strip()
    if path:
        p = Path(path)
        if p.exists() and p.is_dir():
            _local_models_dir = str(p.resolve())
            _save_settings()
            log.info(f"Путь к локальным моделям: {_local_models_dir}")
            return {"status": "ok", "path": _local_models_dir, "found": True}
        else:
            return {"status": "error", "detail": "Папка не найдена"}
    else:
        _local_models_dir = ""
        _save_settings()
        return {"status": "ok", "path": "", "found": False}


@app.post("/api/models/scan")
async def scan_models():
    """Сканирует папку с локальными моделями, ищет доступные."""
    if not _local_models_dir:
        return {"models": []}
    found = []
    models_dir = Path(_local_models_dir)
    for m in AVAILABLE_MODELS:
        model_dir = models_dir / m["id"]
        config = model_dir / "model_config.json"
        ckpt = list(model_dir.glob("*.safetensors"))
        if config.exists() and ckpt:
            found.append({
                "id": m["id"],
                "config": str(config),
                "ckpt": str(ckpt[0]),
                "size_gb": round(ckpt[0].stat().st_size / 1024**3, 2),
            })
    log.info(f"Сканирование локальных моделей: найдено {len(found)}")
    return {"models": found}


@app.post("/api/models/clear-cache")
async def clear_model_cache():
    """Выгружает модель из памяти."""
    global _model_instance
    _model_instance = {"model": None, "model_name": None, "device": None}
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("Модель выгружена из памяти")
    return {"status": "ok"}


# ─── Скачивание модели (фоновое с прогрессом) ────────────────────────────

_downloads = {}       # model_id → status dict
_dl_lock = threading.Lock()

@app.post("/api/models/download/{model_id}")
async def start_download(model_id: str):
    """Запускает фоновое скачивание модели, возвращает сразу."""
    model_info = next((m for m in AVAILABLE_MODELS if m["id"] == model_id), None)
    if not model_info:
        raise HTTPException(status_code=404, detail="Модель не найдена")

    # Проверяем токен
    if not _hf_token:
        raise HTTPException(status_code=401, detail="HF токен не настроен. Сначала введите токен.")

    with _dl_lock:
        existing = _downloads.get(model_id, {})
        if existing.get("status") == "downloading":
            raise HTTPException(status_code=409, detail="Модель уже скачивается")
        # Сразу инициализируем как "downloading" — фронтенд видит прогресс немедленно
        _downloads[model_id] = {
            "status": "downloading",
            "files_total": 0,
            "files_done": 0,
            "total_bytes": 0,
            "downloaded_bytes": 0,
            "percent": 0.0,
            "current_file": "Подготовка…",
            "error": None,
        }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _do_download, model_id, model_info["repo"], _hf_token)
    return {"status": "started", "model_id": model_id}


@app.get("/api/models/download/{model_id}/progress")
async def get_download_progress(model_id: str):
    """Возвращает прогресс скачивания модели."""
    with _dl_lock:
        p = _downloads.get(model_id, {"status": "not_found"})
    return {"model_id": model_id, **p}


def _do_download(model_id: str, repo_id: str, token):
    """Фоновое скачивание: requests.get(stream=True) + HF cache."""
    import time
    import hashlib
    import os
    import shutil
    from pathlib import Path
    import requests

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.constants import HF_HUB_CACHE

        log.info(f"\U0001f680 \u041d\u0430\u0447\u0430\u043b\u043e \u0441\u043a\u0430\u0447\u0438\u0432\u0430\u043d\u0438\u044f {model_id} ({repo_id})")

        # 1. Commit hash + file list
        api = HfApi()

        # Пробуем получить commit hash через repo_info (надёжнее)
        try:
            repo_info = api.repo_info(repo_id, token=token)
            commit_hash = repo_info.sha
            log.info(f"repo_info: commit={commit_hash[:12]}")
        except Exception as e:
            err_str = str(e)
            log.warning(f"repo_info не сработал ({err_str}), пробуем list_repo_refs")
            # Если DNS ошибка — сразу понятное сообщение
            if "getaddrinfo" in err_str or "11001" in err_str:
                raise RuntimeError(
                    f"Не удаётся подключиться к HuggingFace (DNS ошибка).\n"
                    f"  • Проверьте интернет-соединение\n"
                    f"  • Откройте https://huggingface.co в браузере — доступен?\n"
                    f"  • Отключите офлайн-режим в настройках (вкладка Модели)\n"
                    f"  • Если используете VPN/прокси — проверьте настройки"
                )
            if "403" in err_str:
                raise RuntimeError(
                    f"Нет доступа к {repo_id}. Примите лицензию и проверьте токен."
                )
            # Пробуем list_repo_refs как fallback
            try:
                refs_list = api.list_repo_refs(repo_id, token=token)
            except Exception as e2:
                raise RuntimeError(f"Ошибка соединения с HuggingFace: {e2}")
            # В разных версиях huggingface_hub разная структура
            if hasattr(refs_list, 'main'):
                commit_hash = refs_list.main.commit_hash
            elif hasattr(refs_list, 'converted') and refs_list.converted:
                commit_hash = refs_list.converted[0].target_commit
            elif isinstance(refs_list, (list, tuple)):
                commit_hash = refs_list[0].target_commit if refs_list else None
            else:
                commit_hash = None
            if not commit_hash:
                raise RuntimeError("Не удалось получить commit hash для репозитория")

        all_files = api.list_repo_files(repo_id, token=token)
        repo_files = [f for f in all_files if not f.startswith('.') and not f.endswith('.gitignore')]
        log.info(f"Files: {repo_files}")

        # 2. File sizes
        file_info = []
        total_bytes = 0
        for path in repo_files:
            try:
                meta = api.get_paths_info(repo_id, paths=[path], token=token)
                size = meta[0].size if meta and getattr(meta[0], 'size', None) else 0
            except Exception:
                size = 0
            file_info.append({"path": path, "size": size})
            total_bytes += size

        if not file_info:
            raise RuntimeError("No files found to download")

        total_mb = total_bytes / 1_048_576
        log.info(f"{len(file_info)} files, {total_mb:.1f} MB, commit={commit_hash[:8]}")

        # 3. Cache dirs
        cache_root = Path(HF_HUB_CACHE)
        model_cache = cache_root / f"models--{repo_id.replace('/', '--')}"
        blobs_dir = model_cache / "blobs"
        snap_dir = model_cache / "snapshots" / commit_hash
        refs_dir = model_cache / "refs"
        blobs_dir.mkdir(parents=True, exist_ok=True)
        snap_dir.mkdir(parents=True, exist_ok=True)
        refs_dir.mkdir(parents=True, exist_ok=True)
        (refs_dir / "main").write_text(commit_hash + "\n")

        # Existing blobs
        existing_blobs = {
            p.name: p.stat().st_size
            for p in blobs_dir.iterdir()
            if p.is_file() and p.suffix != '.lock' and '.tmp.' not in p.name
        }
        pre_existing_bytes = sum(existing_blobs.values())
        log.info(f"Already in blobs: {len(existing_blobs)} files, {pre_existing_bytes:,} bytes")

        # 4. Init progress
        with _dl_lock:
            _downloads[model_id] = {
                "status": "downloading",
                "files_total": len(file_info),
                "files_done": 0,
                "total_bytes": total_bytes,
                "downloaded_bytes": pre_existing_bytes,
                "percent": round(pre_existing_bytes / total_bytes * 100, 1) if total_bytes > 0 else 0,
                "current_file": "Preparing...",
                "error": None,
            }

        # 5. Auth headers
        dl_headers = {"User-Agent": "stable-audio-3-portable/1.0"}
        if token:
            dl_headers["Authorization"] = f"Bearer {token}"

        # 6. Download each file with stream=True
        downloaded_bytes = pre_existing_bytes

        for fi in file_info:
            path = fi["path"]
            size = fi["size"]

            snap_path = snap_dir / path
            if snap_path.exists():
                log.info(f"Already cached: {path}")
                with _dl_lock:
                    _downloads[model_id]["files_done"] += 1
                continue

            with _dl_lock:
                _downloads[model_id]["current_file"] = path

            url = f"https://huggingface.co/{repo_id}/resolve/main/{path}"
            log.info(f"Downloading {path} ({size:,} bytes)")

            try:
                r = requests.get(url, headers=dl_headers, stream=True, timeout=120)
                r.raise_for_status()

                # Temp file
                rand_suffix = hashlib.sha256(os.urandom(8)).hexdigest()[:12]
                tmp_path = blobs_dir / f".tmp.{rand_suffix}.{path.replace('/', '_')}"

                sha256 = hashlib.sha256()
                file_downloaded = 0

                with open(tmp_path, 'wb') as f_out:
                    for chunk in r.iter_content(chunk_size=65536):
                        if not chunk:
                            continue
                        f_out.write(chunk)
                        sha256.update(chunk)
                        file_downloaded += len(chunk)
                        downloaded_bytes += len(chunk)

                        # Update progress on every chunk
                        with _dl_lock:
                            p = _downloads[model_id]
                            p["downloaded_bytes"] = downloaded_bytes
                            if total_bytes > 0:
                                p["percent"] = round(downloaded_bytes / total_bytes * 100, 1)

                # Rename to blob
                blob_hash = sha256.hexdigest()
                blob_path = blobs_dir / blob_hash

                if blob_path.exists():
                    tmp_path.unlink(missing_ok=True)
                else:
                    tmp_path.rename(blob_path)

                # Create symlink / copy in snapshots
                dest = snap_dir / path
                dest.parent.mkdir(parents=True, exist_ok=True)

                try:
                    os.symlink(blob_path, dest)
                except (OSError, NotImplementedError):
                    shutil.copy2(blob_path, dest)

                with _dl_lock:
                    _downloads[model_id]["files_done"] += 1

                log.info(f"Done {path} - {file_downloaded:,} bytes, blob={blob_hash[:12]}")

            except requests.RequestException as e:
                raise RuntimeError(f"Download error {path}: {e}")

        # Final
        with _dl_lock:
            _downloads[model_id]["status"] = "done"
            _downloads[model_id]["percent"] = 100.0
            _downloads[model_id]["downloaded_bytes"] = total_bytes
            _downloads[model_id]["current_file"] = ""

        log.info(f"Model {model_id} downloaded: {total_bytes:,} bytes")

    except Exception as e:
        err_msg = str(e)
        log.error(f"Download error {model_id}: {err_msg}")
        with _dl_lock:
            _downloads[model_id] = {
                "status": "error",
                "percent": 0.0,
                "files_total": 0, "files_done": 0,
                "total_bytes": 0, "downloaded_bytes": 0,
                "current_file": "",
                "error": err_msg,
            }

@app.post("/api/token")
async def set_token(data: dict = Body(...)):
    """Сохраняет HuggingFace токен."""
    global _hf_token
    token = data.get("token", "").strip()
    if token:
        _hf_token = token
        _save_settings()
        log.info("HuggingFace токен сохранён")
        return {"status": "ok", "configured": True}
    else:
        _hf_token = None
        _save_settings()
        return {"status": "ok", "configured": False}


@app.get("/api/token")
async def get_token_status():
    """Статус токена (без раскрытия значения)."""
    return {"configured": _hf_token is not None and len(_hf_token) > 0}


@app.post("/api/offline")
async def set_offline(data: dict = Body(...)):
    """Включает/выключает офлайн-режим."""
    global _offline_mode
    _offline_mode = bool(data.get("offline", False))
    if _offline_mode:
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
    _save_settings()
    log.info(f"Офлайн-режим: {'вкл' if _offline_mode else 'выкл'}")
    return {"status": "ok", "offline": _offline_mode}


@app.post("/api/output-dir")
async def set_output_dir(data: dict = Body(...)):
    """Устанавливает пользовательскую папку для выходных файлов."""
    global _custom_output_dir, OUTPUT_DIR
    path = data.get("path", "").strip()
    if path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        _custom_output_dir = str(p.resolve())
        OUTPUT_DIR = p.resolve()
    else:
        _custom_output_dir = ""
        OUTPUT_DIR = ROOT_DIR / "output"
        OUTPUT_DIR.mkdir(exist_ok=True)
    _save_settings()
    log.info(f"Папка для выходных файлов: {OUTPUT_DIR}")
    return {"status": "ok", "path": _custom_output_dir or str(OUTPUT_DIR)}


@app.get("/api/output-dir")
async def get_output_dir():
    """Возвращает текущую папку для выходных файлов."""
    return {"path": str(OUTPUT_DIR.resolve())}


@app.post("/api/generate")
async def generate_audio(
    prompt: str = Form(...),
    model_name: str = Form("small-music"),
    duration: float = Form(10.0),
    mode: str = Form("text-to-audio"),
    # Для audio-to-audio
    init_audio: Optional[UploadFile] = File(None),
    init_noise_level: Optional[float] = Form(0.3),
    # Для inpainting
    inpaint_audio: Optional[UploadFile] = File(None),
    inpaint_start: Optional[float] = Form(None),
    inpaint_end: Optional[float] = Form(None),
    # Параметры генерации
    cfg_scale: Optional[float] = Form(7.0),
    steps: Optional[int] = Form(50),
):
    """
    Генерация аудио.
    mode: 'text-to-audio', 'audio-to-audio', 'inpaint'
    """
    if not prompt or prompt.strip() == "":
        raise HTTPException(status_code=400, detail="Промпт не может быть пустым")

    try:
        model = get_model(model_name)

        # Ограничение длительности
        max_dur = {"small-music": 120, "small-sfx": 120, "medium": 380}.get(model_name, 120)
        duration = min(max(duration, 1.0), max_dur)

        # Готовим параметры генерации
        gen_kwargs = {
            "prompt": prompt.strip(),
            "duration": duration,
            "cfg_scale": cfg_scale,
            "steps": steps,
        }

        # Режимы генерации
        if mode == "audio-to-audio" and init_audio:
            init_bytes = await init_audio.read()
            init_ext = Path(init_audio.filename).suffix if init_audio.filename else ".wav"
            gen_kwargs["init_audio"] = _load_audio_from_bytes(init_bytes, init_ext)
            gen_kwargs["init_noise_level"] = init_noise_level or 0.3
            log.info(f"Audio-to-Audio: {init_audio.filename}, noise={init_noise_level}")

        elif mode == "inpaint" and inpaint_audio:
            inpaint_bytes = await inpaint_audio.read()
            inpaint_ext = Path(inpaint_audio.filename).suffix if inpaint_audio.filename else ".wav"
            gen_kwargs["inpaint_audio"] = _load_audio_from_bytes(inpaint_bytes, inpaint_ext)
            if inpaint_start is not None:
                gen_kwargs["inpaint_mask_start_seconds"] = inpaint_start
            if inpaint_end is not None:
                gen_kwargs["inpaint_mask_end_seconds"] = inpaint_end
            log.info(f"Inpaint: audio={inpaint_audio.filename}, mask=[{inpaint_start}-{inpaint_end}]")

        log.info(f"Генерация: model={model_name}, mode={mode}, prompt='{prompt[:60]}...', duration={duration}s")

        # Запускаем генерацию в отдельном потоке
        # generate() возвращает torch.Tensor [batch, channels, samples]
        def _generate():
            return model.generate(**gen_kwargs)

        loop = asyncio.get_event_loop()
        result_tensor = await loop.run_in_executor(None, _generate)

        # Сохраняем WAV
        output_filename = f"output_{uuid.uuid4().hex}.wav"
        output_path = OUTPUT_DIR / output_filename

        import soundfile as sf
        # result_tensor: [B=1, C, T] -> numpy [T, C] or [T]
        audio_np = result_tensor[0].cpu().numpy()  # [C, T]
        if audio_np.shape[0] == 1:
            audio_np = audio_np[0]  # [T] mono
        else:
            audio_np = audio_np.T  # [T, C] stereo

        # Определяем sample rate из модели или используем 44.1kHz
        try:
            sample_rate = int(model.model.sample_rate)
        except (AttributeError, TypeError):
            sample_rate = 44100

        sf.write(str(output_path), audio_np, sample_rate)

        if output_path.exists():
            file_size = output_path.stat().st_size
            log.info(f"Готово: {output_filename} ({file_size/1024:.1f} KB, {duration}s)")
            return {
                "status": "success",
                "filename": output_filename,
                "file_path": f"/outputs/{output_filename}",
                "duration": duration,
                "model": model_name,
                "mode": mode,
                "size_bytes": file_size,
                "sample_rate": sample_rate,
            }
        else:
            raise HTTPException(status_code=500, detail="Файл не был создан")

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Ошибка генерации: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/outputs")
async def list_outputs():
    """Список сгенерированных аудиофайлов."""
    files = []
    for f in sorted(OUTPUT_DIR.glob("*.wav"), key=os.path.getmtime, reverse=True):
        files.append({
            "filename": f.name,
            "size": f.stat().st_size,
            "created": time.ctime(os.path.getmtime(f)),
            "url": f"/outputs/{f.name}",
        })
    return {"outputs": files}


@app.delete("/api/outputs/{filename}")
async def delete_output(filename: str):
    """Удаление аудиофайла."""
    file_path = OUTPUT_DIR / filename
    if file_path.exists():
        file_path.unlink()
        return {"status": "deleted", "filename": filename}
    raise HTTPException(status_code=404, detail="Файл не найден")


@app.get("/outputs/{filename}")
async def serve_output(filename: str):
    """Отдача аудиофайла."""
    file_path = OUTPUT_DIR / filename
    if file_path.exists():
        return FileResponse(str(file_path), media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Файл не найден")


# ─── Статика (SPA) ─────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    """Главная страница GUI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Frontend not found. Run setup first."}


# ─── Запуск ──────────────────────────────────────────────────────────────

def main():
    port = int(os.environ.get("PORT", 8765))
    log.info(f"Stable Audio 3 GUI — http://localhost:{port}")
    import uvicorn
    uvicorn.run(
        "app.server:app",
        host="127.0.0.1",
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
