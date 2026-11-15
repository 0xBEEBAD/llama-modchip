#!/usr/bin/env python3
"""
Builds a 100% natural-UTF8 (Moses-detokenized), benchmark-disjoint multi-domain calibration corpus
(~155 KB, >= 28,000 tokens so --chunks 6 -c 4096 -b 4096 yields 24,576 clean natural BPE tokens).
Strictly verifies 0.0000% 8-gram overlap against wiki-alt.txt, wiki.test.raw, and eval_quality_and_tools.py.
"""
import argparse
import hashlib
import os
import re
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def detokenize_wikitext(s: str) -> str:
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = re.sub(r"\s+([,.;:!?%)\]}])", r"\1", s)
    s = re.sub(r"([(\[{])\s+", r"\1", s)
    s = re.sub(r"\s+('s|'re|'ve|'ll|'d|'m|n't)\b", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"(\b\w+)\s+'\s+", r"\1' ", s)
    # Fix Moses spaced double quotes: " foo bar " -> "foo bar"
    def fix_quotes(line: str) -> str:
        parts = line.split(' " ')
        if len(parts) < 2:
            return line
        out = [parts[0]]
        for idx, p in enumerate(parts[1:]):
            if idx % 2 == 0:
                out.append(' "' + p.lstrip())
            else:
                out.append('" ' + p.lstrip())
        res = "".join(out)
        res = re.sub(r'"\s+([,.;:!?)\]])', r'"\1', res)
        res = re.sub(r'([(\[])\s+"', r'\1"', res)
        return res
    lines = [fix_quotes(ln) for ln in s.split("\n")]
    s = "\n".join(lines)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def norm_words(text: str):
    return re.findall(r"[a-z0-9_]+", text.lower())


def build_ngrams(words, n=8):
    return {tuple(words[i : i + n]) for i in range(0, max(0, len(words) - n + 1))}


def has_overlap(text: str, banned_ngrams: set, n=8) -> bool:
    w = norm_words(text)
    for i in range(0, max(0, len(w) - n + 1)):
        if tuple(w[i : i + n]) in banned_ngrams:
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Build clean natural-UTF8 benchmark-disjoint calibration corpus (24,576 tokens)")
    ap.add_argument("--out", default="./calib_v4_clean_utf8.txt")
    ap.add_argument("--wikitext-zip", default="./wikitext-2-raw-v1.zip")
    ap.add_argument("--llama-dir", default="./llama.cpp")
    ap.add_argument(
        "--bench-files",
        nargs="*",
        default=[
            "./wiki-alt.txt",
            "./wiki.test.raw",
            os.path.join(REPO_ROOT, "benchmarks", "eval_quality_and_tools.py"),
        ],
    )
    args = ap.parse_args()

    banned = set()
    with zipfile.ZipFile(args.wikitext_zip, "r") as zf:
        test_raw = zf.read("wikitext-2-raw/wiki.test.raw").decode("utf-8", errors="ignore")
        train_raw = zf.read("wikitext-2-raw/wiki.train.raw").decode("utf-8", errors="ignore")
    banned |= build_ngrams(norm_words(test_raw), 8)
    n_bench = 1
    for bf in args.bench_files:
        if os.path.isfile(bf):
            txt = open(bf, "r", encoding="utf-8", errors="ignore").read()
            banned |= build_ngrams(norm_words(txt), 8)
            n_bench += 1
    print(f"[Dedup] Indexed {len(banned):,} 8-grams across {n_bench} benchmark sources", flush=True)

    raw_paras = [p.strip() for p in train_raw.split("\n") if len(p.strip()) >= 240 and not p.strip().startswith("=")]
    clean_paras = []
    seen_local = set()
    for p in raw_paras:
        dp = detokenize_wikitext(p)
        if "@" in dp:
            continue
        if has_overlap(dp, banned, 8) or has_overlap(dp, seen_local, 8):
            continue
        seen_local |= build_ngrams(norm_words(dp), 8)
        clean_paras.append(dp)
    print(f"[WikiTrain-CleanUTF8] Retained {len(clean_paras):,} natural UTF-8 disjoint paragraphs", flush=True)

    code_candidates = [
        os.path.join(args.llama_dir, "gguf-py/gguf/gguf_reader.py"),
        os.path.join(args.llama_dir, "gguf-py/gguf/gguf_writer.py"),
        os.path.join(args.llama_dir, "examples/simple/simple.cpp"),
        os.path.join(args.llama_dir, "examples/batched/batched.cpp"),
        os.path.join(args.llama_dir, "src/llama-vocab.cpp"),
    ]
    code_blocks = []
    for cp in code_candidates:
        if os.path.isfile(cp):
            ctxt = open(cp, "r", encoding="utf-8", errors="ignore").read()[:7500]
            if not has_overlap(ctxt, banned, 8) and not has_overlap(ctxt, seen_local, 8):
                seen_local |= build_ngrams(norm_words(ctxt), 8)
                code_blocks.append(f"// File: {os.path.basename(cp)}\n" + ctxt)

    tool_and_math = """
### Structured Function Calling & Systems Reasoning Examples

User: Query the cluster telemetry database for nodes exceeding 78% memory utilization in the last 15 minutes and schedule a rolling drain on the warmest standby pool.
Assistant:
```json
{
  "tool_calls": [
    {
      "id": "call_telemetry_901",
      "type": "function",
      "function": {
        "name": "query_prometheus_range",
        "arguments": "{\\"expr\\": \\"100 * (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) > 78\\", \\"window\\": \\"15m\\", \\"step\\": \\"30s\\"}"
      }
    },
    {
      "id": "call_drain_902",
      "type": "function",
      "function": {
        "name": "plan_rolling_node_drain",
        "arguments": "{\\"pool\\": \\"standby-eu-south\\", \\"max_unavailable\\": 1, \\"respect_pdb\\": true, \\"timeout_sec\\": 600}"
      }
    }
  ]
}
```

### Spectral Approximation & Ridge Regression Theory

Let $H \\in \\mathbb{R}^{N \\times K}$ denote a centered activation matrix over $N$ routed samples and $K$ active channels, with empirical Gram matrix $G = H H^\\top \\in \\mathbb{R}^{N \\times N}$. For any target residual $R \\in \\mathbb{R}^{N \\times D}$, the Tikhonov-regularized minimum-norm linear estimator solves:
$$\\min_{\\Delta W \\in \\mathbb{R}^{D \\times K}} \\| R - H \\Delta W^\\top \\|_F^2 + \\lambda \\| \\Delta W \\|_F^2$$
By the push-through matrix identity $H^\\top (H H^\\top + \\lambda I_N)^{-1} = (H^\\top H + \\lambda I_K)^{-1} H^\\top$, the primal $K \\times K$ ridge system reduces exactly to an $N \\times N$ symmetric positive-definite Cholesky system $(G + \\lambda I_N) Z = R$, followed by $\\Delta W = Z^\\top H$.

### Sintesi Tecnica Multilingue / Synthèse Technique / Technische Zusammenfassung

Nell'architettura dei sistemi distribuiti su memoria unificata DDR5, la latenza di decodifica dipende essenzialmente dalla banda passante effettiva tra i controller di memoria e la gerarchia di cache L2/L3 dei core fisici. Riducendo l'ampiezza intermedia degli esperti sparsi tramite proiezioni ortogonali calibrate sul cono attivo del router, si ottiene simultaneamente una contrazione del traffico DRAM per token e l'eliminazione del rumore di quantizzazione estremo sui proiettori d'uscita.

Dans les réseaux de neurones à mélange d'experts clairsemés, chaque jeton n'active qu'un sous-ensemble restreint de spécialistes. L'analyse spectrale locale restreinte au cône d'activation réel révèle une décroissance rapide des valeurs singulières effectives, permettant de préserver fidèlement la géométrie des représentations latentes tout en allouant davantage de bits aux projections d'attention multi-têtes compressées.

In modernen Mehrkernprozessoren mit breiten Vektoreinheiten wird die Rechendichte bei kleinen Stapelgrößen primär durch die Speicherbandbreite begrenzt. Eine blockweise Sortierung der Spaltennormen verbessert die Homogenität der Skalierungsfaktoren innerhalb jedes Quantisierungsblocks erheblich und reduziert systematische Verzerrungen in tieferen Transformatoren-Schichten.
""".strip()
    assert not has_overlap(tool_and_math, banned, 8)
    seen_local |= build_ngrams(norm_words(tool_and_math), 8)

    def take_bytes(paras, start_idx, target_chars):
        buf = []
        tot = 0
        idx = start_idx
        while idx < len(paras) and tot < target_chars:
            buf.append(paras[idx])
            tot += len(paras[idx]) + 2
            idx += 1
        return "\n\n".join(buf), idx

    # Interleave natural English encyclopedic prose (~78%) + C++/Python code + math/JSON/multilingual (~22%)
    s0, idx = take_bytes(clean_paras, 0, 32000)
    s1, idx = take_bytes(clean_paras, idx, 32000)
    s2_p, idx = take_bytes(clean_paras, idx, 18000)
    s2 = s2_p + "\n\n" + "\n\n".join(code_blocks[:2])
    s3, idx = take_bytes(clean_paras, idx, 32000)
    s4_p, idx = take_bytes(clean_paras, idx, 18000)
    s4 = tool_and_math + "\n\n" + s4_p + "\n\n" + "\n\n".join(code_blocks[2:4])
    s5, idx = take_bytes(clean_paras, idx, 32000)

    full_text = "\n\n".join([s0, s1, s2, s3, s4, s5]) + "\n"
    assert not has_overlap(full_text, banned, 8), "FATAL: 8-gram overlap detected!"
    assert "@-@" not in full_text and "@,@" not in full_text

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(full_text)

    sha = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
    print(f"[OK] Wrote {args.out} ({len(full_text):,} bytes, SHA256={sha})", flush=True)
    print("Sample paragraph 1:", clean_paras[0][:300], flush=True)


if __name__ == "__main__":
    main()
