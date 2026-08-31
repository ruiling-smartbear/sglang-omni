#!/usr/bin/env bash
# Validation benchmark for sglang-omni PR #1840 / issue #975.
# Compares MOSS-Transcribe-Diarize transcription latency on short non-speech
# vs speech audio, baseline (pip sglang-omni==0.1.3) vs the PR branch.
# Intended for a fresh single-GPU environment (A10G/L4 class is enough).
set -u

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

echo "== install baseline sglang-omni==0.1.3 =="
pip install -q "sglang-omni==0.1.3" 2>&1 | tail -2
# the package pins typer>=0.9.0, but the sgl-omni CLI uses typing.Literal
# options that older typer can't parse; make sure a current typer is present.
pip install -q -U typer 2>&1 | tail -1

echo "== fetch test audio =="
rm -rf /tmp/so && git clone -q --depth 1 --filter=blob:none --sparse https://github.com/sgl-project/sglang-omni /tmp/so
cd /tmp/so && git sparse-checkout set -q tests/data

python3 - <<'PY'
import soundfile as sf, numpy as np
d, sr = sf.read('/tmp/so/tests/data/cough.wav')
if d.ndim > 1:
    d = d.mean(axis=1)
target = int(6.0 * sr)
reps = max(1, -(-target // len(d)))
sf.write('/tmp/nonspeech6s.wav', np.tile(d, reps)[:target], sr)
print('made /tmp/nonspeech6s.wav: 6.0s at', sr, 'Hz (looped cough = non-speech)')
PY

echo "== fetch ESC-50 non-speech clips (laughing x3, crying_baby, sneezing) =="
curl -sL https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv -o /tmp/esc50.csv
python3 - <<'PY'
import csv, os, urllib.request
rows = list(csv.DictReader(open('/tmp/esc50.csv')))
def pick(cat, n): return [r['filename'] for r in rows if r['category'] == cat][:n]
picks = pick('laughing', 3) + pick('crying_baby', 1) + pick('sneezing', 1)
os.makedirs('/tmp/esc', exist_ok=True)
for i, f in enumerate(picks):
    cat = next(r['category'] for r in rows if r['filename'] == f)
    dst = f'/tmp/esc/{i}_{cat}.wav'
    urllib.request.urlretrieve(f'https://github.com/karolpiczak/ESC-50/raw/master/audio/{f}', dst)
    print('fetched', f, '->', dst)
PY

# DeepGEMM's _C.so dlopens libnvrtc.so.13, which lives inside the pip-installed
# nvidia wheels rather than on the system loader path; expose those dirs and
# skip DeepGEMM JIT anyway (not needed for a bf16 0.9B model).
export SGL_ENABLE_JIT_DEEPGEMM=0
NVIDIA_LIBS=$(python3 -c "import glob;print(':'.join(sorted(set(glob.glob('/usr/local/lib/python3.*/site-packages/nvidia/*/lib')))))")
export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"
python3 -c "import ctypes,glob;f=glob.glob('/usr/local/lib/python3.*/site-packages/nvidia/*/lib/libnvrtc.so.13');print('libnvrtc.so.13 found:',bool(f));f and ctypes.CDLL(f[0])" || pip install -q "nvidia-cuda-nvrtc-cu13" 2>&1 | tail -1

serve() {
  nohup sgl-omni serve --model-path OpenMOSS-Team/MOSS-Transcribe-Diarize \
    --port 8000 --mem-fraction-static 0.80 --torch-compile off > "/tmp/server_$1.log" 2>&1 &
  echo $! > /tmp/server.pid
}

wait_ready() {
  for i in $(seq 1 200); do
    curl -sf -o /dev/null localhost:8000/health && { echo "server ready after ~$((i*5))s"; return 0; }
    sleep 5
  done
  echo "SERVER FAILED TO START"; tail -40 "/tmp/server_$1.log"; return 1
}

stop_server() {
  kill "$(cat /tmp/server.pid)" 2>/dev/null; sleep 8
  pkill -f "sgl-omni" 2>/dev/null; sleep 4
}

bench() {  # $1 = label
  for f in /tmp/nonspeech6s.wav /tmp/so/tests/data/cough.wav /tmp/so/tests/data/query_to_cars.wav /tmp/esc/*.wav; do
    for i in 1 2 3; do
      T=$(curl -s -o /tmp/resp.json -w '%{time_total}' -X POST localhost:8000/v1/audio/transcriptions \
        -F model=OpenMOSS-Team/MOSS-Transcribe-Diarize -F "file=@$f" -F response_format=json)
      TXT=$(python3 -c "import json;t=json.load(open('/tmp/resp.json')).get('text','');print(f'chars={len(t)} | '+t[:60].replace(chr(10),' '))" 2>/dev/null || head -c 60 /tmp/resp.json)
      echo "RESULT | $1 | $(basename "$f") | run$i | ${T}s | ${TXT}"
    done
  done
}

echo "== BASELINE 0.1.3 =="
serve baseline
if ! wait_ready baseline; then echo "ABORT: baseline server failed"; exit 1; fi
bench "baseline-0.1.3"
stop_server

echo "== install PR #1840 branch =="
pip install -q "git+https://github.com/ruiling-smartbear/sglang-omni.git@fix/moss-td-short-audio-token-budget" 2>&1 | tail -2
pip install -q -U typer 2>&1 | tail -1

echo "== FIXED PR#1840 =="
serve fixed
if ! wait_ready fixed; then echo "ABORT: fixed server failed"; exit 1; fi
bench "pr1840"
stop_server

echo "BENCH_DONE"
