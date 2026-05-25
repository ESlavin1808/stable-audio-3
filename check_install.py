"""Check Stable Audio 3 installation"""
import sys, importlib

packages = [
    ('torch', 'torch'),
    ('stable_audio_3', 'stable_audio_3'),
    ('soundfile', 'soundfile'),
    ('fastapi', 'fastapi'),
    ('uvicorn', 'uvicorn'),
    ('numpy', 'numpy'),
]

ok = True
for name, mod_name in packages:
    try:
        mod = importlib.import_module(mod_name)
        ver = getattr(mod, '__version__', '?')
        print(f'  [OK] {name} == {ver}')
    except ImportError as e:
        print(f'  [FAIL] {name}: {e}')
        ok = False

# Torch CUDA check
import torch
print(f'')
print(f'  Torch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    print(f'  VRAM: {vram / 1e9:.1f} GB')
    print(f'  CUDA capability: {props.major}.{props.minor}')

if ok:
    print(f'\n  => All OK! Start with: run_portable.bat')
else:
    print(f'\n  => Some packages missing')
