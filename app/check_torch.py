"""
Check torch + CUDA status. Returns exit code:
  0 = torch with CUDA works -> keep it
  1 = torch exists but CPU only -> needs reinstall
  2 = torch not installed -> needs install
"""
import subprocess, sys, os

try:
    import torch
    ver = torch.__version__
    cuda = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda else "N/A"
    cc = torch.cuda.get_device_capability(0) if cuda else None
    driver = "?"
    try:
        r = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        for line in r.stdout.split('\n'):
            if 'Driver Version' in line:
                driver = line.split('Driver Version:')[1].strip().split()[0]
                break
    except: pass

    print(f"  Torch: {ver}, CUDA: {cuda}")
    if cuda:
        print(f"  GPU: {gpu} (CC {cc[0]}.{cc[1]}), Драйвер: {driver}")
        sys.exit(0)  # keep it
    else:
        sys.exit(1)  # needs CUDA reinstall
except ImportError:
    print("  Torch: не установлен")
    sys.exit(2)
