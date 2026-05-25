"""Start Stable Audio 3 server in background"""
import subprocess, sys, os, time, signal

cwd = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(cwd, 'venv', 'Scripts', 'python.exe')
log_file = os.path.join(cwd, 'server.log')

# Перенаправляем кэш HuggingFace в папку программы
os.environ.setdefault('HF_HOME', os.path.join(cwd, 'hf_cache'))
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', os.path.join(cwd, 'hf_cache', 'hub'))

# Создаём папки, если нет
for d in [os.path.join(cwd, 'hf_cache'), os.path.join(cwd, 'hf_cache', 'hub'),
          os.path.join(cwd, 'output'), os.path.join(cwd, 'models')]:
    os.makedirs(d, exist_ok=True)

log_handle = open(log_file, 'w', encoding='utf-8')

proc = subprocess.Popen(
    [venv_python, '-m', 'uvicorn', 'app.server:app', '--host', '127.0.0.1', '--port', '8765'],
    cwd=cwd,
    stdout=log_handle,
    stderr=subprocess.STDOUT,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
)

print(f'Server PID: {proc.pid}')
print(f'Log: {log_file}')
print(f'URL: http://localhost:8765')

# Wait a bit and check
time.sleep(3)

import urllib.request, json
try:
    resp = urllib.request.urlopen('http://localhost:8765/api/device', timeout=5)
    data = json.loads(resp.read())
    print(f'GPU: {data.get("gpu_name", "N/A")}')
    print(f'CUDA: {data.get("cuda_available")}')
    print('Server is RUNNING!')
except Exception as e:
    print(f'Check log for errors: {e}')
    log_handle.close()
    with open(log_file, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[-20:]:
            print(f'  LOG: {line.rstrip()}')
