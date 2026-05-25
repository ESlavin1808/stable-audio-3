"""
Check torch + CUDA status. Returns exit code:
  0 = torch with CUDA works and GPU is compatible -> keep it
  1 = torch exists but GPU incompatible or CPU only -> needs reinstall
  2 = torch not installed -> needs install
"""
import subprocess, sys, os

try:
    import torch
    ver = torch.__version__
    cuda = torch.cuda.is_available()
    gpu_name = "?"
    cc = (0, 0)
    driver = "?"

    if cuda:
        gpu_name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
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
        print(f"  GPU: {gpu_name} (CC {cc[0]}.{cc[1]}), Драйвер: {driver}")

        # Check if GPU is actually supported by this torch build
        # cu126 supports sm_50-sm_90, cu128 supports sm_50-sm_120
        supported = torch.cuda.get_arch_list() if hasattr(torch.cuda, 'get_arch_list') else []
        if supported:
            arch_str = f"sm_{cc[0]}{cc[1]}"
            is_compat = any(arch_str in s for s in supported)
            if not is_compat:
                print(f"  [WARNING] GPU (CC {cc[0]}.{cc[1]}) НЕ поддерживается текущим torch {ver}")
                print(f"  Поддерживаемые: {supported}")
                print(f"  Нужен torch 2.10+ с cu128 (Blackwell/RTX 50xx)")
                sys.exit(1)  # needs reinstall with cu128

        sys.exit(0)  # all good, keep it
    else:
        print(f"  GPU: {gpu_name} (без CUDA)")
        sys.exit(1)  # needs CUDA reinstall
except ImportError:
    print("  Torch: не установлен")
    sys.exit(2)
