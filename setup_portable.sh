#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/venv"

echo "================================================"
echo "  Stable Audio 3 — Portable Edition"
echo "================================================"
echo ""

# ─── Detect OS ────────────────────────────────────────
OS="$(uname -s)"
case "$OS" in
    Linux*)   OS_TYPE="linux";;
    Darwin*)  OS_TYPE="macos";;
    *)
        echo "[ERROR] Unsupported OS: $OS"
        echo "  This script supports Linux and macOS."
        echo "  Windows users: use setup_portable.bat"
        exit 1
        ;;
esac
echo "[OS] $OS_TYPE detected"

# ─── Check uv ─────────────────────────────────────────
if ! command -v uv &> /dev/null; then
    echo "[ERROR] uv not found."
    echo "  Install: pip install uv"
    echo "  Or: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# ─── Virtual environment ──────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "[1] Virtual environment already exists"
else
    echo "[1] Creating virtual environment..."
    uv venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ─── Check torch + CUDA/MPS ──────────────────────────
echo ""
echo "[2] Checking existing Torch..."
NEED_TORCH=false
python "$ROOT_DIR/app/check_torch.py" && NEED_TORCH=false || NEED_TORCH=true

if [ "$NEED_TORCH" = true ]; then
    echo ""
    echo "  Installing Torch..."

    if [ "$OS_TYPE" = "macos" ]; then
        # macOS: try MPS (Apple Silicon) or CPU
        echo "  Attempt: macOS (MPS / Apple Silicon)"
        uv pip install torch torchvision torchaudio --force-reinstall
    elif [ "$OS_TYPE" = "linux" ]; then
        # Linux: try cu128 -> cu126 -> CPU
        echo "  Attempt 1: cu128 (Blackwell / RTX 50xx)"
        if uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall; then
            python "$ROOT_DIR/app/check_torch.py" || {
                echo "  Attempt 2: cu126 (Ampere / RTX 30-40xx)"
                uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
                python "$ROOT_DIR/app/check_torch.py" || {
                    echo "  [WARNING] CUDA didn't work, installing CPU version..."
                    uv pip install torch torchvision torchaudio
                }
            }
        else
            echo "  cu128 failed, trying cu126..."
            uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
            python "$ROOT_DIR/app/check_torch.py" || {
                echo "  [WARNING] CUDA didn't work, installing CPU version..."
                uv pip install torch torchvision torchaudio
            }
        fi
    fi
else
    echo "  [OK] Torch with GPU works and is compatible -- skipping."
fi

# ─── stable-audio-3 (without deps, torch already there) ──
echo ""
echo "[3] Installing stable-audio-3..."
uv pip install "git+https://github.com/Stability-AI/stable-audio-3.git" --no-deps

# ─── stable-audio-3 dependencies (except torch) ──────
echo ""
echo "[3b] Installing stable-audio-3 dependencies..."
uv pip install einops einops-exts huggingface-hub numpy safetensors soundfile tqdm transformers

# ─── Web server ─────────────────────────────────────
echo ""
echo "[4] Installing web server and utilities..."
uv pip install fastapi uvicorn requests python-multipart

# ─── Verify ─────────────────────────────────────────
echo ""
echo "=== Verification ==="
python "$ROOT_DIR/app/check_torch.py"
python -c "import stable_audio_3; print('  stable-audio-3: OK')" 2>/dev/null || echo "  [WARNING] stable-audio-3 check failed"
python -c "import fastapi; import uvicorn; import requests; print('  fastapi/uvicorn/requests: OK')"

echo ""
echo "================================================"
echo "  Installation complete!"
echo "  Run: ./run_portable.sh"
echo "================================================"
