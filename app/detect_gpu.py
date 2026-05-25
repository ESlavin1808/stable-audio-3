"""
Detect GPU + CUDA version. Informational only.
Recommends the right PyTorch index URL based on GPU compute capability.
"""
import subprocess
import sys
import re
import os
from pathlib import Path


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
            if line.strip().startswith('|') and 'NVIDIA' in line and not line.startswith('|   N/A'):
                parts = line.split('|')
                if len(parts) >= 2:
                    name_raw = parts[1].strip()
                    m = re.match(r'\d+\s+(NVIDIA\s+\S+(?:\s+\S+)*?)(?:\s{2,}|$)', name_raw)
                    if m:
                        gpu_name = m.group(1).strip()

            m = re.search(r'CUDA Version:\s*(\d+\.\d+)', line)
            if m:
                cuda_version = m.group(1)
            m = re.search(r'Driver Version:\s*(\d+\.\d+)', line)
            if m:
                driver_version = m.group(1)

        return gpu_name, cuda_version, driver_version
    except FileNotFoundError:
        return None, None, None
    except subprocess.TimeoutExpired:
        return None, None, None
    except Exception as e:
        return None, None, str(e)


def get_torch_index(device_capability=None):
    """
    Pick the right PyTorch index URL based on GPU compute capability.
    cu126: supports sm_50-sm_90 (older GPUs up to RTX 40xx)
    cu128: supports sm_50-sm_120 (includes RTX 50xx Blackwell)
    """
    if device_capability:
        major, minor = device_capability
        # CC 9.0+ (Hopper/Blackwell) need torch with sm_90+ support
        if (major >= 9 and minor >= 0) or major >= 10:
            return "cu128", "https://download.pytorch.org/whl/cu128"
        # CC 8.0+ (Ampere) work with both, cu128 preferred
        elif major >= 8:
            return "cu128", "https://download.pytorch.org/whl/cu128"
    
    # Default or older GPUs: cu126 (widest compatibility)
    # Try cu128 first, fall back to cu126
    return "cu128", "https://download.pytorch.org/whl/cu128"


def print_gpu_summary():
    """Prints a human-readable GPU summary."""
    gpu, cuda, driver = get_nvidia_info()
    torch_label = "не установлен"
    
    try:
        import torch
        torch_label = f"{torch.__version__}"
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            torch_label += f" (CC {cc[0]}.{cc[1]})"
    except ImportError:
        pass

    lines = []
    lines.append("=" * 50)
    lines.append("  GPU и CUDA")
    lines.append("=" * 50)

    if gpu:
        lines.append(f"  GPU:        {gpu}")
    else:
        lines.append(f"  GPU:        Не обнаружена")

    if cuda:
        lines.append(f"  CUDA:       {cuda}")
    else:
        lines.append(f"  CUDA:       Не обнаружена")

    if driver:
        lines.append(f"  Драйвер:    {driver}")
    
    lines.append(f"  Torch:      {torch_label}")
    
    # Recommend
    try:
        import torch
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            tag, url = get_torch_index(cc)
            lines.append(f"  Рекомендуемый индекс: {tag} ({url})")
    except ImportError:
        tag, url = get_torch_index()
        lines.append(f"  Рекомендуемый индекс: {tag} ({url})")

    lines.append("=" * 50)
    return "\n".join(lines)


if __name__ == '__main__':
    print()
    print(print_gpu_summary())
    print()
