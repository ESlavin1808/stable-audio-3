"""
Detect GPU + CUDA version. Informational only.
Shown at startup in run_portable.bat.
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
            # GPU name from the GPU table row
            if line.strip().startswith('|') and 'NVIDIA' in line and not line.startswith('|   N/A'):
                parts = line.split('|')
                if len(parts) >= 2:
                    name_raw = parts[1].strip()
                    m = re.match(r'\d+\s+(NVIDIA\s+\S+(?:\s+\S+)*?)(?:\s{2,}|$)', name_raw)
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
        return None, None, None
    except subprocess.TimeoutExpired:
        return None, None, None
    except Exception as e:
        return None, None, str(e)


def print_gpu_summary():
    """Prints a human-readable GPU summary."""
    gpu, cuda, driver = get_nvidia_info()
    import torch

    lines = []
    lines.append("=" * 50)
    lines.append("  >> GPU и CUDA <<")
    lines.append("=" * 50)

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

    lines.append(f"  Torch:      {torch.__version__}")
    lines.append(f"  CUDA in     Torch: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        lines.append(f"  GPU name:   {torch.cuda.get_device_name(0)}")

    lines.append("=" * 50)
    return "\n".join(lines)


if __name__ == '__main__':
    print()
    print(print_gpu_summary())
    print()
