#!/usr/bin/env bash
# run-dsv4.sh — DeepSeek-V4-Flash-0731-Fractal (284B, 56.56 GiB) on an Apple Silicon Mac with 64 GB.
#
# Build the engine first with ./build-dsv4-engine.sh (unified llama-modchip v3.0:
# llama.cpp de32b268c + patches/llama-modchip-v3.0.patch). Stock llama.cpp cannot load
# FRACTAL_64 (44) / VQ3_64 (45) GGUFs.
#
# PROFILE=fast    (default) experts of 5 pure-IQ1_S layers on the CPU. ~51.8 GB on the GPU, so it needs
#                 `sudo sysctl iogpu.wired_limit_mb=57344` (resets at reboot). Measured on an M5 Max:
#                 llama-bench tg128 27.0 tok/s, pp512 169; llama-server 20.7-23.6 tok/s, 0 swapouts.
# PROFILE=nosudo  experts of 8 pure-IQ1_S layers on the CPU. ~49 GB on the GPU, which fits the default
#                 Metal limit. llama-bench tg128 23.7 tok/s.
#
# Do not use --n-cpu-moe: it takes layers 0..N-1, and with those on the CPU decode is 13.6 tok/s
# (--n-cpu-moe 4) instead of 27.0. Why is not established; see results/dsv4-m5max.md.
# Speculative decoding (ngram-mod) is slower on this model.
#
# Environment: LLAMA_DIR, MODEL, PORT (8789), CTX (32768), THREADS (12; measured best, 18 drops to 4.1 tok/s).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${LLAMA_DIR:-}" ]]; then
    if [[ -x "$HERE/../workspace/llama.cpp-v3/build/bin/llama-server" ]]; then
        LLAMA_DIR="$HERE/../workspace/llama.cpp-v3/build"
    elif [[ -x "$HERE/../llama.cpp/build/bin/llama-server" ]]; then
        LLAMA_DIR="$HERE/../llama.cpp/build"
    else
        LLAMA_DIR="$HERE/../workspace/llama.cpp-dsv4/build"
    fi
fi
if [[ -z "${MODEL:-}" ]]; then
    if [[ -f "$HOME/models/dsv4/DeepSeek-V4-Flash-0731-MaxQ-v4-00001-of-00003.gguf" ]]; then
        MODEL="$HOME/models/dsv4/DeepSeek-V4-Flash-0731-MaxQ-v4-00001-of-00003.gguf"
    elif [[ -f "$HERE/../workspace/models/dsv4-maxq-v4/DeepSeek-V4-Flash-0731-MaxQ-v4-00001-of-00003.gguf" ]]; then
        MODEL="$HERE/../workspace/models/dsv4-maxq-v4/DeepSeek-V4-Flash-0731-MaxQ-v4-00001-of-00003.gguf"
    else
        MODEL="$HOME/models/dsv4/DeepSeek-V4-Flash-0731-Fractal-00001-of-00003.gguf"
    fi
fi
PORT="${PORT:-8789}"
CTX="${CTX:-32768}"
THREADS="${THREADS:-12}"
PROFILE="${PROFILE:-fast}"

case "$PROFILE" in
    fast)   cpu_layers='10|12|14|16|18'; need_mb=57344 ;;
    nosudo) cpu_layers='10|12|14|16|18|20|22|24'; need_mb=0 ;;
    *) echo "Unknown PROFILE=$PROFILE (use fast or nosudo)" >&2; exit 1 ;;
esac

if [[ ! -f "$MODEL" ]]; then echo "Model not found: $MODEL" >&2; exit 1; fi
if [[ ! -x "$LLAMA_DIR/bin/llama-server" ]]; then echo "llama-server not found in $LLAMA_DIR/bin" >&2; exit 1; fi

limit="$(sysctl -n iogpu.wired_limit_mb)"
if (( need_mb > 0 && limit < need_mb )); then
    echo "PROFILE=fast needs iogpu.wired_limit_mb >= $need_mb (now $limit)." >&2
    echo "Run: sudo sysctl iogpu.wired_limit_mb=$need_mb   or use PROFILE=nosudo" >&2
    exit 1
fi

echo "[run-dsv4] profile=$PROFILE cpu_expert_layers=$cpu_layers ctx=$CTX threads=$THREADS port=$PORT" >&2
echo "[run-dsv4] if the log shows kIOGPUCommandBufferCallbackErrorOutOfMemory, output is invalid: stop and use PROFILE=nosudo" >&2
export RC_TABLES_DIR="/nonexistent"
exec "$LLAMA_DIR/bin/llama-server" \
    --model "$MODEL" \
    --alias deepseek-v4-flash \
    --host 127.0.0.1 --port "$PORT" \
    --cors-origins localhost \
    --ctx-size "$CTX" \
    --flash-attn on \
    --n-gpu-layers 99 \
    --override-tensor "blk\.(${cpu_layers})\.ffn_.*_exps\.=CPU" \
    --fit off \
    --parallel 1 \
    --jinja \
    --threads "$THREADS" \
    "$@"
