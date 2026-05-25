"""
Detect GPU + CUDA version and recommend PyTorch install command.
Used by setup_portable.bat to auto-configure dependencies.
"""
import subprocess
import sys
import re
import os


def get_nvidia_info():
    """Try to detect NVIDIA GPU and CUDA version via nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True, text=True, timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
        )
        if result.returncode != 0:
            return None, None, None

        output = result.stdout
        gpu_name = None
        cuda_version = None
        driver_version = None

        for line in output.split('\n'):
            # GPU name from the GPU table row: "|   0  NVIDIA GeForce RTX 5060 Ti   WDDM  |"
            if line.strip().startswith('|') and 'NVIDIA' in line and not line.startswith('|   N/A'):
                parts = line.split('|')
                if len(parts) >= 2:
                    name_raw = parts[1].strip()
                    # Extract the model name before the first double-space or WDDM/Tesla/etc
                    m = re.match(r'\d+\s+(NVIDIA\s+\S+(?:\s+\S+)*?)(?:\s{2,}|$)', name_raw)
                    if m:
                        gpu_name = m.group(1).strip()

            # Also try the AMD/Intel path
            if not gpu_name and ('AMD' in line or 'Radeon' in line or 'Intel' in line or 'Arc' in line):
                parts = line.split('|')
                if len(parts) >= 2:
                    name_raw = parts[1].strip()
                    m = re.match(r'\d+\s+(.*?)(?:\s{2,}|$)', name_raw)
                    if m:
                        gpu_name = m.group(1).strip()

            # CUDA Version
            m = re.search(r'CUDA Version:\s*(\d+\.\d+)', line)
            if m:
                cuda_version = m.group(1)

            # Driver Version
            m = re.search(r'Driver Version:\s*(\d+\.\d+)', line)
            if m:
                driver_version = m.group(1)

        return gpu_name, cuda_version, driver_version

    except FileNotFoundError:
        return None, None, None  # nvidia-smi not found
    except subprocess.TimeoutExpired:
        return None, None, None
    except Exception as e:
        return None, None, str(e)


def get_pytorch_url(cuda_version_str):
    """
    Map CUDA version to PyTorch index URL + pip package spec.
    Returns (torch_spec, index_url) or (cpu_spec, None).
    """
    if cuda_version_str is None:
        return ("torch torchaudio", None)  # CPU

    try:
        cuda_ver = float(cuda_version_str)
    except ValueError:
        return ("torch torchaudio", None)

    # Map CUDA version → PyTorch index URL
    # PyTorch 2.7.x wheels available: cu126, cu124, cu118
    if cuda_ver >= 12.4:
        return ("torch==2.7.1 torchaudio==2.7.1", "https://download.pytorch.org/whl/cu126")
    elif cuda_ver >= 12.1:
        return ("torch==2.7.1 torchaudio==2.7.1", "https://download.pytorch.org/whl/cu124")
    elif cuda_ver >= 11.8:
        return ("torch==2.5.1 torchaudio==2.5.1", "https://download.pytorch.org/whl/cu118")
    else:
        return ("torch torchaudio", None)  # CPU fallback


def get_gpu_summary():
    """Returns a human-readable GPU summary and install recommendations."""
    gpu, cuda, driver = get_nvidia_info()

    lines = []
    lines.append("=" * 55)
    lines.append("  >> Обнаружение GPU и CUDA <<")
    lines.append("=" * 55)

    if gpu:
        lines.append(f"  GPU:        {gpu}")
    else:
        lines.append(f"  GPU:        Не обнаружена (NVIDIA)")

    if cuda:
        lines.append(f"  CUDA:       {cuda}")
    else:
        lines.append(f"  CUDA:       Не обнаружена")

    if driver:
        lines.append(f"  Драйвер:    {driver}")
    else:
        lines.append(f"  Драйвер:    nvidia-smi не найден")

    if gpu:
        spec, url = get_pytorch_url(cuda)
        if url:
            lines.append(f"  >> [OK] PyTorch {spec} (CUDA {cuda})")
        else:
            lines.append(f"  >> CUDA {cuda} не поддерживается, установка CPU-версии")
    else:
        lines.append(f"  >> NVIDIA GPU не найдена -- установка CPU-версии PyTorch")
        lines.append(f"  >> Будет работать, но генерация будет медленной")

    lines.append("=" * 55)
    return "\n".join(lines)


def install_pytorch(venv_python):
    """Run pip install with the correct PyTorch package."""
    import subprocess

    gpu, cuda, driver = get_nvidia_info()
    spec, url = get_pytorch_url(cuda)

    if url:
        cmd = [venv_python, '-m', 'pip', 'install', *spec.split(),
               '--index-url', url, '--force-reinstall']
        print(f"  Установка: {' '.join(cmd[:6])} --index-url {url}")
    else:
        cmd = [venv_python, '-m', 'pip', 'install', *spec.split()]
        print(f"  Установка (CPU): {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"  [ОШИБКА] pip install: возврат {result.returncode}")
        print(f"  Пробуем CPU-версию...")
        cmd2 = [venv_python, '-m', 'pip', 'install', 'torch', 'torchaudio',
                '--force-reinstall']
        result2 = subprocess.run(cmd2, capture_output=False, text=True)
        if result2.returncode != 0:
            print(f"  [ОШИБКА] CPU тоже не встал: возврат {result2.returncode}")
            return False

    # Verify
    verify = subprocess.run(
        [venv_python, '-c', 'import torch; print(torch.__version__); print(torch.cuda.is_available())'],
        capture_output=True, text=True, timeout=30,
    )
    print(f"  Torch: {verify.stdout.strip()}")
    if verify.returncode != 0:
        print(f"  [ОШИБКА] torch не импортируется: {verify.stderr[:200]}")
        return False
    return True


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--install':
        if len(sys.argv) > 2:
            install_pytorch(sys.argv[2])
        else:
            print("  Использование: python detect_gpu.py --install <path_to_python>")
    elif len(sys.argv) > 1 and sys.argv[1] == '--recommend':
        # Output one-line pip install string for setup_portable.bat
        gpu, cuda, driver = get_nvidia_info()
        spec, url = get_pytorch_url(cuda)
        if url:
            print(f'"{spec}" --index-url {url}')
        else:
            print(f'"{spec}"')
    else:
        # Show info
        print(get_gpu_summary())
        print()
