<div align="center">

# 🎵 Stable Audio 3 — Portable GUI

**Локальный генератор музыки и звуковых эффектов**
<br>
*Stable Audio 3 с веб-интерфейсом — полностью портативная версия*

</div>

---

## 📋 О проекте

Портативная сборка **Stable Audio 3** от Stability AI с графическим интерфейсом.  
Всё необходимое — в одной папке. Не требует установки в систему.

### Возможности

| Функция | Описание |
|---------|----------|
| 🎵 **Генерация музыки** | Три модели: Small Music (быстрая), Small SFX (эффекты), Medium (качественная) |
| 🎛️ **Три режима** | Текст-в-аудио, аудио-в-аудио, инпейнтинг (замена фрагмента) |
| 📦 **Портативность** | Всё в папке программы — зависимости, модели, настройки, аудио |
| 🌐 **Веб-интерфейс** | Открывается в браузере, тёмная тема, русский/английский |
| 🔌 **Локальная работа** | Полный офлайн-режим после загрузки моделей |
| 🖥️ **GPU ускорение** | CUDA, автоматический выбор устройства |

---

## ⚙️ Системные требования

| Компонент | Требование |
|-----------|-----------|
| **OS** | Windows 10/11, Linux, macOS (Intel / Apple Silicon) |
| **Python** | 3.10+ (авто-определение через uv) |
| **GPU** | NVIDIA с ≥8 ГБ VRAM (рекомендуется) или CPU (медленно) |
| **Место** | ~2 ГБ на зависимости + ~5 ГБ на каждую модель |
| **Инструменты** | [`uv`](https://docs.astral.sh/uv/#installation) (установщик пакетов) |

### Зачем uv?

`uv` в 10-100x быстрее pip и корректно обрабатывает окружения Python 3.10.

```shell
# Установка uv (один раз)
pip install uv
```

---

## 🚀 Быстрый старт

### 1. Установка зависимостей

**Windows:**
```batch
setup_portable.bat
```

**Linux / macOS:**
```bash
chmod +x setup_portable.sh
./setup_portable.sh
```

Что делает скрипт:
- Создаёт виртуальное окружение Python 3.10
- Устанавливает PyTorch с CUDA 12.6 (или CPU fallback)
- Устанавливает Stable Audio 3 из исходников GitHub
- Устанавливает FastAPI + Uvicorn (веб-сервер)
- Создаёт папки: `output/`, `hf_cache/`

> **Модели не скачиваются** — это делается из интерфейса.

### 2. Получение доступа к моделям

Модели Stable Audio 3 — **gated** (требуют принятия лицензии):

1. Зарегистрируйтесь на [huggingface.co](https://huggingface.co)
2. Примите лицензии на страницах моделей (кнопка *Agree and access repository*):
   - [stabilityai/stable-audio-3-small-music](https://huggingface.co/stabilityai/stable-audio-3-small-music)
   - [stabilityai/stable-audio-3-small-sfx](https://huggingface.co/stabilityai/stable-audio-3-small-sfx)
   - [stabilityai/stable-audio-3-medium](https://huggingface.co/stabilityai/stable-audio-3-medium)
3. Создайте токен: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Тип токена: **Read** (достаточно для скачивания)

### 3. Запуск

**Windows:**
```batch
run_portable.bat
```

**Linux / macOS:**
```bash
./run_portable.sh
```

После запуска откройте браузер по адресу [http://localhost:8765](http://localhost:8765)

### 4. Загрузка модели (из интерфейса)

1. Откройте вкладку **«Модели»**
2. Введите **HF-токен** (из шага 2)
3. Нажмите **«Скачать»** на нужной модели
4. Наблюдайте прогресс в реальном времени
5. После завершения офлайн-режим включится автоматически

---

## 📁 Структура проекта

```
stable-audio-3/                  # Можно скопировать на флешку
├── app/
│   ├── server.py               # FastAPI-бэкенд
│   ├── __init__.py
│   └── static/
│       └── index.html          # Веб-интерфейс (SPA)
├── venv/                       # Виртуальное окружение (создаётся setup)
├── hf_cache/                   # Кэш HuggingFace (модели) [авто]
│   └── hub/                    #   └── стандартная структура HF
├── models/                     # Локальные модели (опционально)
├── output/                     # Сгенерированные аудиофайлы
├── settings.json               # Настройки (создаётся при первом запуске)
├── settings.example.json       # Пример настроек
├── setup_portable.bat          # Установка зависимостей (Windows)
├── setup_portable.sh           # Установка зависимостей (Linux/macOS)
├── run_portable.bat            # Запуск (Windows)
├── run_portable.sh             # Запуск (Linux/macOS)
├── start_server.py             # Скрипт запуска сервера
├── check_install.py            # Проверка установки
└── README.md
```

### Портативность

Проект **полностью портативный**:
- `venv/` — изолированный Python со всеми зависимостями
- `hf_cache/` — модели кэшируются в папке программы
- `models/` — можно подключить локально скачанные модели
- `output/` — сгенерированные файлы

> Перенесите папку `stable-audio-3/` на другой компьютер — всё будет работать (после принятия лицензий на HF).

---

## 🗂️ Управление моделями

### Загрузка из HuggingFace

После ввода токена можно скачать любую из трёх моделей:

| Модель | ID | Параметры | Длительность |
|--------|----|-----------|-------------|
| Small Music | `small-music` | 433M | до 120 с |
| Small SFX | `small-sfx` | 433M | до 120 с |
| Medium | `medium` | 1.4B | до 380 с (GPU) |

Прогресс загрузки отображается в реальном времени на каждый скачанный блок.

### Локальные модели (без HuggingFace)

Если модели уже скачаны на диск:

1. Укажите путь в поле **«Путь к локальным моделям»**
2. Структура папок по ID модели:
   ```
   models/
   ├── small-music/
   │   ├── model_config.json
   │   └── model.safetensors
   ├── small-sfx/
   │   ├── model_config.json
   │   └── model.safetensors
   └── medium/
       ├── model_config.json
       └── model.safetensors
   ```

### Офлайн-режим

После загрузки модели через HuggingFace кэшируются в `hf_cache/` и работают **без интернета**.  
Офлайн-режим включается автоматически после успешной загрузки.

---

## ⚙️ Настройки

Файл `settings.json` создаётся автоматически при первом запуске:

```json
{
  "offline": false,           // Офлайн-режим
  "hf_token": "",             // HF-токен (скрыт в интерфейсе)
  "local_models_dir": "",     // Путь к локальным моделям
  "output_dir": ""            // Куда сохранять аудио (пусто = ./output/)
}
```

> `settings.json` добавлен в `.gitignore` — токен не попадёт в репозиторий.

---

## 🔧 Решение проблем

| Проблема | Решение |
|----------|---------|
| **`uv` не найден** | Установите: `pip install uv` или `curl -LsSf https://astral.sh/uv/install.sh | sh` |
| **403 при скачивании модели** | Примите лицензию на HF и/или проверьте токен |
| **CUDA не найдена (Linux)** | Проверьте драйвер NVIDIA (≥525), переустановите PyTorch |
| **MPS не работает (macOS)** | Убедитесь, что Python собран для arm64, используйте `python3` вместо `python` |
| **Out of memory** | Используйте small-модель, уменьшите `steps` или длительность |
| **Медленная генерация** | Убедитесь, что используется GPU (вкладка «Модели») |
| **Порт 8765 занят** | Закройте другой процесс или измените порт в `start_server.py` |
| **Ошибка `No module named...`** | Запустите `setup_portable.bat` заново |
| **После закрытия окна процесс висит** | Убейте процесс `python.exe` через Диспетчер задач |

---

## 🔬 Технические детали

- **Бэкенд:** FastAPI + Uvicorn (Python)
- **Фронтенд:** HTML + CSS + JS (SPA, без фреймворков, тёмная тема)
- **Аудио-движок:** [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3)
- **Формат:** 44.1 kHz stereo WAV
- **Кэш-менеджмент:** HuggingFace Hub (перенаправлен в `hf_cache/`)
- **Загрузка:** `requests.get(stream=True)` с прогрессом на каждый 64KB чанк

### Структура HF-кэша

```
hf_cache/
└── hub/
    ├── models--stabilityai--stable-audio-3-small-music/
    │   ├── blobs/          # Файлы моделей по SHA256
    │   ├── refs/           # Ссылки на коммиты
    │   └── snapshots/      # Симлинки на blobs
    ├── models--stabilityai--stable-audio-3-small-sfx/
    └── models--stabilityai--stable-audio-3-medium/
```

---

## 📄 Лицензии

**Код проекта** — MIT License. Делайте с ним что хотите.

**Модели Stable Audio 3** имеют собственную лицензию Stability AI.  
Для скачивания необходимо принять лицензию на HuggingFace (разово, для каждой модели).

---

<div align="center">

**Сделано с ❤️ для локального творчества**

</div>
