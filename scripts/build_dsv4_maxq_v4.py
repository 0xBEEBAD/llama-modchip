#!/usr/bin/env python3
"""
build_dsv4_maxq_v4.py — 100% Hermetic Donor-Free Builder for DeepSeek-V4-Flash-0731-MaxQ-v4 (< 58.00 GiB)

100% Donor-Free Provenance:
  1. Upstream weights: deepseek-ai/DeepSeek-V4-Flash-0731 @ 7872f01b1d1fe23eabc4c98b48bffcef5a386062
  2. Public routed-MoE imatrix: antirez/deepseek-v4-gguf @ f71f23d552d664e523b422157b2befbf74040380
     (imatrix/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat, SHA256: 02a7c78c29875e46... — collected on chat-v2)
  3. Benchmark-disjoint natural UTF-8 multi-domain calibration: calib_v4_clean (24,576 tokens = 6 x 4096 ctx +
     exact Top-6 routed expert IDs + imatrix_v4_clean_24k.dat; 0.0000% >=8-gram overlap with wiki-alt.txt)
  4. Local single-pass MaxQ-Fractal GGUF (built from 0731 @ 7872f01b + @antirez 1.5M imatrix via build_dsv4_maxq_singlepass.py)

Exact Pareto-Dominant Precision & Width Allocation (57.993 GiB < 58.000 GiB = 62,277,025,792 B; ZERO custom Q2_0/VQ3_0 types):
  - ZERO K=1536 layers:
      * 9 full-width anchor layers (K=2048, uncentered ridge, no bias): {0, 30, 34, 36, 37, 39, 40, 41, 42}
      * 34 cone-collapsed layers (K=1792 = 7 x 256, -12.5% tail, post-IQ2_XXS shrunk residual centroid bias 0.60 * mu)
  - MoE precision & DualGram ridge compensation (NC=384, Deterministic Routed-Stride + Router Sqrt-Softplus Top-Score Fill):
      * L42 ffn_{gate,up}_exps -> IQ2_XXS (2.06 bpw, fresh BETA_0731=0.50), ffn_down_exps -> MSE-optimal MXFP4 (4.25 bpw)
      * L41 ffn_{gate,up}_exps -> IQ1_M   (1.75 bpw, fresh BETA_0731=0.50), ffn_down_exps -> MSE-optimal MXFP4 (4.25 bpw)
      * L00..L40 ffn_down_exps -> norm-sorted IQ2_XXS (2.06 bpw, BETA_HD=0.80) + FP64 Woodbury DualGram (lam_mul=4.5 / 6.0)
      * L20, L26..L40 ffn_{gate,up}_exps -> fresh IQ1_M (1.75 bpw, BETA_0731=0.50); L01, L04 -> fresh IQ1_S;
        remaining 23 IQ1_S layers -> reused single-pass MaxQ-Fractal slices reordered by 0731 ||Wd[:,j]||^2
  - Dense / MLA / Compressor / Indexer precision (all 446 K-quant 2D tensors weighted with imatrix_v4_clean_24k.dat):
      * 43x attn_kv.weight -> Q8_0 (8.50 bpw)
      * 145x KV/Indexer compressor & indexer q_b projections -> Q6_K (6.56 bpw)
      * L00 MLA (attn_q_b, attn_output_a, attn_output_b) -> Q5_K; L42 MLA -> Q6_K; L01..L41 MLA -> Q4_K
"""
import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import mmap
import os
import re
import struct
import threading
import time
import urllib.error
import urllib.request

ROW_BYTES = {
    0:  lambda c: c * 4,
    2:  lambda c: (c // 32) * 18,
    8:  lambda c: (c // 32) * 34,
    12: lambda c: (c // 256) * 144,
    13: lambda c: (c // 256) * 176,
    14: lambda c: (c // 256) * 210,
    16: lambda c: (c // 256) * 66,
    17: lambda c: (c // 256) * 74,
    18: lambda c: (c // 256) * 98,
    19: lambda c: (c // 256) * 50,
    22: lambda c: (c // 256) * 82,
    26: lambda c: c * 4,
    29: lambda c: (c // 256) * 56,
    30: lambda c: c * 2,
    39: lambda c: (c // 32) * 17,
    42: lambda c: (c // 64) * 30,
}

DTYPE_CODE = {"F32": 0, "BF16": 1, "F8_E4M3": 2, "I64": 3, "I32": 4}

K_2048_LAYERS  = {0, 30, 34, 36, 37, 39, 40, 41, 42}
IQ2_GU_LAYERS  = {42}
IQ1M_GU_LAYERS = {20} | set(range(26, 42))

ROOT_MAP = {
    "token_embd.weight":      "embed.weight",
    "output_norm.weight":     "norm.weight",
    "output.weight":          "head.weight",
    "output_hc_base.weight":  "hc_head_base",
    "output_hc_fn.weight":    "hc_head_fn",
    "output_hc_scale.weight": "hc_head_scale",
}

LAYER_MAP = {
    "attn_compressor_ape.weight":      "attn.compressor.ape",
    "attn_compressor_gate.weight":     "attn.compressor.wgate.weight",
    "attn_compressor_kv.weight":       "attn.compressor.wkv.weight",
    "attn_compressor_norm.weight":     "attn.compressor.norm.weight",
    "attn_kv.weight":                  "attn.wkv.weight",
    "attn_kv_a_norm.weight":           "attn.kv_norm.weight",
    "attn_norm.weight":                "attn_norm.weight",
    "attn_output_a.weight":            "attn.wo_a.weight",
    "attn_output_b.weight":            "attn.wo_b.weight",
    "attn_q_a.weight":                 "attn.wq_a.weight",
    "attn_q_a_norm.weight":            "attn.q_norm.weight",
    "attn_q_b.weight":                 "attn.wq_b.weight",
    "attn_sinks.weight":               "attn.attn_sink",
    "exp_probs_b.bias":                "ffn.gate.bias",
    "ffn_down_shexp.weight":           "ffn.shared_experts.w2.weight",
    "ffn_gate_inp.weight":             "ffn.gate.weight",
    "ffn_gate_shexp.weight":           "ffn.shared_experts.w1.weight",
    "ffn_gate_tid2eid.weight":         "ffn.gate.tid2eid",
    "ffn_norm.weight":                 "ffn_norm.weight",
    "ffn_up_shexp.weight":             "ffn.shared_experts.w3.weight",
    "hc_attn_base.weight":             "hc_attn_base",
    "hc_attn_fn.weight":               "hc_attn_fn",
    "hc_attn_scale.weight":            "hc_attn_scale",
    "hc_ffn_base.weight":              "hc_ffn_base",
    "hc_ffn_fn.weight":                "hc_ffn_fn",
    "hc_ffn_scale.weight":             "hc_ffn_scale",
    "indexer.attn_q_b.weight":         "attn.indexer.wq_b.weight",
    "indexer.proj.weight":             "attn.indexer.weights_proj.weight",
    "indexer_compressor_ape.weight":   "attn.indexer.compressor.ape",
    "indexer_compressor_gate.weight":  "attn.indexer.compressor.wgate.weight",
    "indexer_compressor_kv.weight":    "attn.indexer.compressor.wkv.weight",
    "indexer_compressor_norm.weight":  "attn.indexer.compressor.norm.weight",
}


def k_for_layer(bid: int) -> int:
    return 2048 if bid in K_2048_LAYERS else 1792


def dense_target_ttype(bid: int, suf: str, old_tt: int) -> int:
    if suf == "attn_kv.weight":
        return 8
    if suf in ("attn_q_b.weight", "attn_output_a.weight", "attn_output_b.weight"):
        if bid == 0:
            return 13
        if bid == 42:
            return 14
        return 12
    if suf in (
        "attn_compressor_gate.weight",
        "attn_compressor_kv.weight",
        "indexer.attn_q_b.weight",
        "indexer_compressor_gate.weight",
        "indexer_compressor_kv.weight",
    ):
        return 14
    assert old_tt != 42, f"Unexpected Fractal-64 tensor blk.{bid}.{suf}"
    return old_tt


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NO_REDIR_OPENER = urllib.request.build_opener(_NoRedirect)


def resolve_cdn_and_size(hf_url: str) -> tuple[str, int]:
    for attempt in range(1, 8):
        try:
            req = urllib.request.Request(hf_url, method="HEAD", headers={"User-Agent": "llama-modchip/2.5.5"})
            try:
                resp = _NO_REDIR_OPENER.open(req, timeout=20)
                sz = int(resp.headers["Content-Length"])
                resp.close()
                return hf_url, sz
            except urllib.error.HTTPError as e:
                if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
                    loc = e.headers["Location"]
                    with urllib.request.urlopen(
                        urllib.request.Request(loc, method="HEAD", headers={"User-Agent": "llama-modchip/2.5.5"}),
                        timeout=20,
                    ) as r2:
                        return loc, int(r2.headers["Content-Length"])
                raise
        except Exception as exc:
            if attempt == 7:
                raise
            print(f"[CDN] Retry {attempt}/7 for {os.path.basename(hf_url)}: {exc}", flush=True)
            time.sleep(2.0 * attempt)
    raise RuntimeError("unreachable")


def safetensor_is_complete(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        sz = os.path.getsize(path)
        if sz < 16:
            return False
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            if 8 + hlen > sz:
                return False
            hdr = json.loads(f.read(hlen).decode("utf-8"))
        max_end = 0
        for k, v in hdr.items():
            if k != "__metadata__":
                max_end = max(max_end, v["data_offsets"][1])
        return (8 + hlen + max_end) == sz
    except Exception:
        return False


def download_safetensor_parallel(dl_pool: concurrent.futures.ThreadPoolExecutor, hf_url: str, dst_path: str) -> None:
    if safetensor_is_complete(dst_path):
        return
    part_path = dst_path + ".part"
    for attempt in range(1, 6):
        try:
            cdn_url, total_sz = resolve_cdn_and_size(hf_url)
            fd = os.open(part_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
            try:
                os.ftruncate(fd, total_sz)

                def _dl_slice(start: int, length: int) -> None:
                    end_incl = start + length - 1
                    for sub in range(1, 7):
                        cu = cdn_url if sub == 1 else resolve_cdn_and_size(hf_url)[0]
                        req = urllib.request.Request(
                            cu,
                            headers={"Range": f"bytes={start}-{end_incl}", "User-Agent": "llama-modchip/2.5.5"},
                        )
                        try:
                            with urllib.request.urlopen(req, timeout=60) as r:
                                pos = 0
                                while pos < length:
                                    chunk = r.read(min(4 * 1024 * 1024, length - pos))
                                    if not chunk:
                                        break
                                    os.pwrite(fd, chunk, start + pos)
                                    pos += len(chunk)
                                if pos == length:
                                    return
                                raise RuntimeError(f"short read {pos} != {length}")
                        except Exception:
                            if sub == 6:
                                raise
                            time.sleep(1.2 * sub)

                step = 64 * 1024 * 1024
                futs = [dl_pool.submit(_dl_slice, off, min(step, total_sz - off)) for off in range(0, total_sz, step)]
                for f in futs:
                    f.result()
                os.fsync(fd)
            finally:
                os.close(fd)
            if not safetensor_is_complete(part_path):
                raise RuntimeError(f"Safetensor completeness check failed for {part_path}")
            os.replace(part_path, dst_path)
            return
        except Exception as exc:
            if attempt == 5:
                raise
            print(f"[Shard-DL] Retry {attempt}/5 for {os.path.basename(dst_path)}: {exc}", flush=True)
            time.sleep(2.0 * attempt)


class SafetensorReader:
    def __init__(self, path: str):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        self.size = os.path.getsize(path)
        self.mm = mmap.mmap(self.fd, self.size, access=mmap.ACCESS_COPY)
        hlen = struct.unpack_from("<Q", self.mm, 0)[0]
        self.hdr = json.loads(self.mm[8 : 8 + hlen].decode("utf-8"))
        self.data_base = 8 + hlen
        self._cbuf = (ctypes.c_uint8 * self.size).from_buffer(self.mm)
        self.base_addr = ctypes.addressof(self._cbuf)

    def get_info(self, name: str):
        meta = self.hdr[name]
        s, e = meta["data_offsets"]
        return meta["dtype"], meta["shape"], self.base_addr + self.data_base + s, e - s

    def close(self):
        try:
            os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except Exception:
            pass
        del self._cbuf
        self.mm.close()
        os.close(self.fd)


def encode_gguf_str(s: str) -> bytes:
    sb = s.encode("utf-8")
    return struct.pack("<Q", len(sb)) + sb


def parse_gguf_header(buf: bytes):
    off = 24
    magic, ver, nt, nkv = struct.unpack_from("<4sIQQ", buf, 0)
    assert magic == b"GGUF" and ver == 3
    sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def rs():
        nonlocal off
        n = struct.unpack_from("<Q", buf, off)[0]
        off += 8
        s = buf[off : off + n].decode("utf-8", errors="replace")
        off += n
        return s

    def sv(vt):
        nonlocal off
        if vt in sz:
            off += sz[vt]
        elif vt == 8:
            rs()
        elif vt == 9:
            at, an = struct.unpack_from("<IQ", buf, off)
            off += 12
            if at in sz:
                off += sz[at] * an
            else:
                for _ in range(an):
                    sv(at)
        else:
            raise ValueError(vt)

    for _ in range(nkv):
        rs()
        vt = struct.unpack_from("<I", buf, off)[0]
        off += 4
        sv(vt)
    kv_end = off

    tensors = []
    for _ in range(nt):
        name = rs()
        nd = struct.unpack_from("<I", buf, off)[0]
        off += 4
        dims = list(struct.unpack_from(f"<{nd}Q", buf, off))
        off += 8 * nd
        tt, toff = struct.unpack_from("<IQ", buf, off)
        off += 12
        ncol = dims[0]
        nrow = 1
        for d in dims[1:]:
            nrow *= d
        tensors.append({
            "name": name,
            "dims": dims,
            "ttype": tt,
            "off": toff,
            "ncol": ncol,
            "nrow": nrow,
            "nbytes": nrow * ROW_BYTES[tt](ncol),
        })
    dstart = (off + 31) // 32 * 32
    tensors.sort(key=lambda x: x["off"])
    return bytes(buf[:kv_end]), dstart, tensors


def rewrite_kv_prefix(
    src_kv_bytes: bytes,
    new_nt: int,
    total_split_tensors: int,
    str_overrides: dict = None,
    u32_overrides: dict = None,
    str_arr_overrides: dict = None,
    extra_u32_arr: dict = None,
) -> bytearray:
    magic, ver, _old_nt, nkv = struct.unpack_from("<4sIQQ", src_kv_bytes, 0)
    assert magic == b"GGUF"
    off = 24
    sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def skip_val(vt, pos):
        if vt in sz:
            return pos + sz[vt]
        if vt == 8:
            sl = struct.unpack_from("<Q", src_kv_bytes, pos)[0]
            return pos + 8 + sl
        if vt == 9:
            at, an = struct.unpack_from("<IQ", src_kv_bytes, pos)
            pos += 12
            if at in sz:
                return pos + sz[at] * an
            for _ in range(an):
                pos = skip_val(at, pos)
            return pos
        raise ValueError(vt)

    seen_keys = set()
    entries_buf = bytearray()
    saw_split_cnt = False
    for _ in range(nkv):
        kl = struct.unpack_from("<Q", src_kv_bytes, off)[0]
        key = src_kv_bytes[off + 8 : off + 8 + kl].decode("utf-8")
        vt = struct.unpack_from("<I", src_kv_bytes, off + 8 + kl)[0]
        vs = off + 8 + kl + 4
        ve = skip_val(vt, vs)
        seen_keys.add(key)
        entries_buf.extend(encode_gguf_str(key))
        entries_buf.extend(struct.pack("<I", vt))
        if str_overrides and key in str_overrides and vt == 8:
            entries_buf.extend(encode_gguf_str(str_overrides[key]))
        elif u32_overrides and key in u32_overrides and vt in (4, 5):
            entries_buf.extend(struct.pack("<I" if vt == 4 else "<i", int(u32_overrides[key])))
        elif str_arr_overrides and key in str_arr_overrides and vt == 9:
            vals = str_arr_overrides[key]
            entries_buf.extend(struct.pack("<IQ", 8, len(vals)))
            for s in vals:
                entries_buf.extend(encode_gguf_str(s))
        elif extra_u32_arr and key in extra_u32_arr and vt == 9:
            vals = extra_u32_arr[key]
            entries_buf.extend(struct.pack("<IQ", 4, len(vals)))
            for v in vals:
                entries_buf.extend(struct.pack("<I", int(v)))
        elif key == "split.tensors.count":
            saw_split_cnt = True
            if vt == 5:
                entries_buf.extend(struct.pack("<i", total_split_tensors))
            elif vt == 4:
                entries_buf.extend(struct.pack("<I", total_split_tensors))
            else:
                raise ValueError(vt)
        else:
            entries_buf.extend(src_kv_bytes[vs:ve])
        off = ve
    assert saw_split_cnt

    extra_cnt = 0
    if extra_u32_arr:
        for k, vals in extra_u32_arr.items():
            if k not in seen_keys:
                extra_cnt += 1
                entries_buf.extend(encode_gguf_str(k))
                entries_buf.extend(struct.pack("<I", 9))
                entries_buf.extend(struct.pack("<IQ", 4, len(vals)))
                for v in vals:
                    entries_buf.extend(struct.pack("<I", int(v)))

    out = bytearray(struct.pack("<4sIQQ", b"GGUF", ver, new_nt, nkv + extra_cnt))
    out.extend(entries_buf)
    return out


def rewrite_shard1(src_s1_path: str, dst_s1_path: str, total_split_tensors: int) -> int:
    src_bytes = open(src_s1_path, "rb").read()
    magic, _ver, nt, _nkv = struct.unpack_from("<4sIQQ", src_bytes, 0)
    assert magic == b"GGUF" and nt == 0
    overrides = {
        "general.name": "DeepSeek-V4-Flash-0731-MaxQ-v4-64GB",
        "general.quantized_by": "Synchopate (llama-modchip v2.5.5, 100% donor-free 0731 ConeCollapse + DualGram)",
        "general.repo_url": "https://github.com/synchopate/llama-modchip",
        "general.base_model.0.name": "DeepSeek-V4-Flash-0731",
        "general.base_model.0.organization": "DeepSeek",
        "general.base_model.0.repo_url": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731",
        "quantize.imatrix.file": "antirez/deepseek-v4-gguf@f71f23d5:imatrix/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat + imatrix_v4_clean_24k.dat",
        "quantize.imatrix.dataset": "antirez ds4 routed-moe 1.5M tokens (chat-v2, 1 chunk) + disjoint clean UTF-8 24,576-token 0731 calibration (6x4096 ctx, --process-output)",
    }
    u32_ov = {
        "quantize.imatrix.entries_count": 898,
        "quantize.imatrix.chunks_count": 7,
    }
    str_arr_ov = {
        "general.tags": ["llama-modchip", "deepseek-v4", "cpu-inference", "conecollapse", "maxq-v4", "64gb"],
    }
    extra_arr = {
        "deepseek4.expert_feed_forward_lengths": [2048 if i in K_2048_LAYERS else 1792 for i in range(43)],
    }
    out = rewrite_kv_prefix(
        src_bytes,
        new_nt=0,
        total_split_tensors=total_split_tensors,
        str_overrides=overrides,
        u32_overrides=u32_ov,
        str_arr_overrides=str_arr_ov,
        extra_u32_arr=extra_arr,
    )
    pad = ((len(out) + 31) // 32 * 32) - len(out)
    out.extend(b"\x00" * pad)
    tmp_path = dst_s1_path + ".tmp_s1"
    with open(tmp_path, "wb") as f:
        f.write(out)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, dst_s1_path)
    return len(out)


def load_imatrix_ctypes(path: str, lib) -> dict:
    im = {}
    with open(path, "rb") as f:
        n_entries = struct.unpack("<i", f.read(4))[0]
        for _ in range(n_entries):
            slen = struct.unpack("<i", f.read(4))[0]
            name = f.read(slen).decode("utf-8")
            ncall, nval = struct.unpack("<ii", f.read(8))
            raw = f.read(nval * 4)
            arr = (ctypes.c_float * nval).from_buffer_copy(raw)
            if ncall > 0 and ncall != 1:
                lib.mq4_scale_f32(arr, nval, ctypes.c_float(1.0 / float(ncall)))
            im[name] = arr
    return im


def write_all(fd: int, mv: memoryview, total_len: int) -> None:
    pos = 0
    step = 64 * 1024 * 1024
    while pos < total_len:
        pos += os.write(fd, mv[pos : min(pos + step, total_len)])


def copy_local_range(src_fd: int, src_off: int, dst_fd: int, nbytes: int, copy_buf: bytearray) -> None:
    mv = memoryview(copy_buf)
    rem = nbytes
    cur = src_off
    while rem > 0:
        to_read = min(len(copy_buf), rem)
        got = os.preadv(src_fd, [mv[:to_read]], cur) if hasattr(os, "preadv") else 0
        if got <= 0:
            chunk = os.pread(src_fd, to_read, cur)
            if not chunk:
                raise RuntimeError(f"Unexpected EOF in local GGUF at offset {cur}")
            os.write(dst_fd, chunk)
            got = len(chunk)
        else:
            write_all(dst_fd, mv, got)
        cur += got
        rem -= got


def extract_df_cache(v4a_prefix: str, pack_prefix: str) -> None:
    """Extract 100% donor-free pre-quantized 2D Q8_0/Q6_K tensors and K=2048 identity-order gate/up layers from v4a."""
    pack_bin = f"{pack_prefix}.pack"
    pack_idx = f"{pack_prefix}.json"
    copy_buf = bytearray(64 * 1024 * 1024)
    entries_2d = {}
    entries_gu = []
    cur_off = 0

    out_fd = os.open(pack_bin, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
    try:
        for s_idx in ("00002", "00003"):
            p = f"{v4a_prefix}-{s_idx}-of-00003.gguf"
            if not os.path.isfile(p):
                continue
            sfd = os.open(p, os.O_RDONLY)
            try:
                hdr = os.pread(sfd, 8 * 1024 * 1024, 0)
                _, dstart, ts = parse_gguf_header(hdr)
                for t in ts:
                    nm = t["name"]
                    m = re.match(r"blk\.(\d+)\.(.+)$", nm)
                    if not m:
                        continue
                    bid = int(m.group(1))
                    suf = m.group(2)
                    tt = t["ttype"]
                    nb = t["nbytes"]
                    # Keep ONLY Q8_0 (tt==8) unweighted 2D tensors; K-quants MUST be quantized with --imatrix-2d
                    if suf not in ("ffn_down_exps.weight", "ffn_down_exps.bias", "ffn_gate_exps.weight", "ffn_up_exps.weight"):
                        want_tt = dense_target_ttype(bid, suf, -1)
                        if tt == want_tt and tt == 8:
                            copy_local_range(sfd, dstart + t["off"], out_fd, nb, copy_buf)
                            entries_2d[nm] = {"tt": tt, "off": cur_off, "nb": nb}
                            cur_off += nb
                    # Keep ONLY MXFP4-down (bid in {41, 42}) identity-order gate/up slices;
                    # NEVER extract IQ2_XXS-down K=2048 layers (L30,34,36,37,39) because their rows are already col_ord-permuted!
                    elif suf in ("ffn_gate_exps.weight", "ffn_up_exps.weight") and t["dims"] == [4096, 2048, 256]:
                        if bid == 42 and tt == 16:
                            kind = "gate" if "gate" in suf else "up"
                            copy_local_range(sfd, dstart + t["off"], out_fd, nb, copy_buf)
                            entries_gu.append({"bid": bid, "kind": kind, "tt": tt, "off": cur_off, "nb": nb})
                            cur_off += nb
            finally:
                os.close(sfd)
        os.fsync(out_fd)
    finally:
        os.close(out_fd)

    with open(pack_idx, "w", encoding="utf-8") as f:
        json.dump({"2d": entries_2d, "gu": entries_gu}, f)
    print(
        f"[DF-Cache] Extracted {len(entries_2d)} 2D tensors + {len(entries_gu)} K=2048 MoE slices "
        f"({cur_off / (1024**3):.2f} GiB) -> {pack_bin}",
        flush=True,
    )


ANTIREZ_IMATRIX_URL = (
    "https://huggingface.co/antirez/deepseek-v4-gguf/resolve/"
    "f71f23d552d664e523b422157b2befbf74040380/imatrix/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat"
)
ANTIREZ_IMATRIX_SHA256 = "02a7c78c29875e4653d6ce21d8821c02161e83ed90c506bdd8d275f76d4ac97e"


def ensure_antirez_imatrix(path: str) -> str:
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_p = path + ".part"
    print(f"[Imatrix] Downloading pinned @antirez routed-moe-ds4-1p5m.dat -> {path}...", flush=True)
    req = urllib.request.Request(ANTIREZ_IMATRIX_URL, headers={"User-Agent": "llama-modchip/2.5.5"})
    h = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_p, "wb") as out:
        while True:
            chunk = resp.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            out.write(chunk)
    if h.hexdigest() != ANTIREZ_IMATRIX_SHA256:
        os.unlink(tmp_p)
        raise RuntimeError(f"SHA-256 mismatch on routed-moe-ds4-1p5m.dat: {h.hexdigest()}")
    os.replace(tmp_p, path)
    return path


def ensure_st_index(path: str, hf_base: str) -> str:
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    url = f"{hf_base.rstrip('/')}/model.safetensors.index.json"
    print(f"[Index] Fetching {url} -> {path}...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "llama-modchip/2.5.5"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(path, "wb") as out:
        out.write(resp.read())
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-df-cache", nargs=2, metavar=("V4A_PREFIX", "PACK_PREFIX"))
    ap.add_argument("--df-cache-prefix", default="")
    ap.add_argument("--base-gguf-prefix", default=os.environ.get("MAXQ_BASE_GGUF_PREFIX", "./DeepSeek-V4-Flash-0731-MaxQ-Fractal"))
    ap.add_argument("--hf-base", default="https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/resolve/7872f01b1d1fe23eabc4c98b48bffcef5a386062")
    ap.add_argument("--st-index", default=os.environ.get("MAXQ_ST_INDEX", "./model.safetensors.index.json"))
    ap.add_argument("--shards-dir", default=os.environ.get("MAXQ_SHARDS_DIR", "./_sp_shards"))
    ap.add_argument("--imatrix", default=os.environ.get("MAXQ_ANTIREZ_IMATRIX", "./routed-moe-ds4-1p5m.dat"))
    ap.add_argument("--imatrix-2d", default=os.environ.get("MAXQ_IMATRIX_2D", "./imatrix_v4_clean_24k.dat"))
    ap.add_argument("--allow-unweighted-2d", action="store_true")
    ap.add_argument("--calib-dir", default=os.environ.get("MAXQ_CALIB_DIR", "./calib_v4_clean"))
    ap.add_argument("--n-cal-tokens", type=int, default=24576)
    ap.add_argument("--lib-path", default="./workspace/bin/libmaxq_v4_collapse.so")
    ap.add_argument("--out-prefix", default="")
    ap.add_argument("--max-buffered-shards", type=int, default=5)
    ap.add_argument("--dl-workers", type=int, default=16)
    ap.add_argument("--omp-threads", type=int, default=16)
    ap.add_argument("--output-norm-tau", type=float, default=1.00000)
    ap.add_argument("--rewrite-shard1-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.extract_df_cache:
        extract_df_cache(args.extract_df_cache[0], args.extract_df_cache[1])
        return

    if args.rewrite_shard1_only:
        assert args.out_prefix, "--out-prefix required with --rewrite-shard1-only"
        s1_src = f"{args.base_gguf_prefix}-00001-of-00003.gguf"
        s1_dst = f"{args.out_prefix}-00001-of-00003.gguf"
        sz = rewrite_shard1(s1_src, s1_dst, 1362)
        print(f"[Shard 00001] Rewrote {s1_dst} ({sz:,} B, 1,362 split tensors)", flush=True)
        return

    assert args.lib_path and args.out_prefix, "--lib-path and --out-prefix required"
    if not args.dry_run and not args.allow_unweighted_2d:
        if not args.imatrix_2d or not os.path.isfile(args.imatrix_2d):
            raise RuntimeError(
                f"--imatrix-2d file not found ({args.imatrix_2d!r}); required for MaxQ-v4 2D K-quant parity "
                "(pass --allow-unweighted-2d only for diagnostic ablations)."
            )
    t_start = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    os.makedirs(args.shards_dir, exist_ok=True)

    exp_bf16_sz = args.n_cal_tokens * 4096 * 2
    exp_ids_sz  = args.n_cal_tokens * 6
    if not args.dry_run:
        for bid in range(43):
            bf_p = os.path.join(args.calib_dir, f"dsv4_calib_L{bid:02d}.bf16")
            id_p = os.path.join(args.calib_dir, f"dsv4_ids_L{bid:02d}.u8")
            assert os.path.isfile(bf_p) and os.path.getsize(bf_p) >= exp_bf16_sz, f"Bad calib BF16 {bf_p}"
            assert os.path.isfile(id_p) and os.path.getsize(id_p) >= exp_ids_sz,  f"Bad calib IDs {id_p}"
        ensure_antirez_imatrix(args.imatrix)

    ensure_st_index(args.st_index, args.hf_base)
    weight_map = json.load(open(args.st_index, "r", encoding="utf-8"))["weight_map"]

    lib = ctypes.CDLL(args.lib_path)
    lib.mq4_init.argtypes = [ctypes.c_int]
    lib.mq4_init.restype = None
    lib.mq4_scale_f32.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_float]
    lib.mq4_scale_f32.restype = None
    lib.mq4_quantize_2d.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_void_p, ctypes.c_int64,
    ]
    lib.mq4_quantize_2d.restype = ctypes.c_int
    lib.mq4_collapse_moe_layer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.mq4_collapse_moe_layer.restype = ctypes.c_float
    lib.mq4_init(args.omp_threads)

    df_cache_2d = {}
    df_pack_path = ""
    base_preq_gu = {}  # (bid, "gate"|"up", ttype) -> (path, abs_off, nbytes)

    if args.df_cache_prefix and os.path.isfile(f"{args.df_cache_prefix}.json") and os.path.isfile(f"{args.df_cache_prefix}.pack"):
        df_pack_path = f"{args.df_cache_prefix}.pack"
        cmeta = json.load(open(f"{args.df_cache_prefix}.json", "r", encoding="utf-8"))
        df_cache_2d = cmeta.get("2d", {})
        for g in cmeta.get("gu", []):
            g_bid = int(g["bid"])
            if g_bid not in (41, 42):
                raise RuntimeError(
                    f"Refusing df_cache MoE slice for L{g_bid:02d} (down_dst_type != 39): "
                    "non-MXFP4 down layers are col_ord-permuted and cannot be re-permuted!"
                )
            base_preq_gu[(g_bid, g["kind"], int(g["tt"]))] = (df_pack_path, int(g["off"]), int(g["nb"]))

    raw_shards = []
    gate_inp_loc = {}
    probs_b_loc  = {}
    total_split_tensors = 0

    for s_idx in ("00002", "00003"):
        base_path = f"{args.base_gguf_prefix}-{s_idx}-of-00003.gguf"
        with open(base_path, "rb") as f:
            peek = f.read(8 * 1024 * 1024)
        kv_prefix, old_dstart, base_tensors = parse_gguf_header(peek)
        bmap = {t["name"]: t for t in base_tensors}

        for bid in range(43):
            for kind, suf in (("gate", "ffn_gate_exps.weight"), ("up", "ffn_up_exps.weight")):
                kn = f"blk.{bid}.{suf}"
                if kn in bmap and bmap[kn]["ttype"] == 19:
                    kt = bmap[kn]
                    exp_b = 256 * 2048 * 16 * 50
                    assert kt["nbytes"] == exp_b
                    base_preq_gu.setdefault((bid, kind, 19), (base_path, old_dstart + kt["off"], kt["nbytes"]))
            if bid >= 3:
                gn = f"blk.{bid}.ffn_gate_inp.weight"
                bn = f"blk.{bid}.exp_probs_b.bias"
                if gn in bmap:
                    gt = bmap[gn]
                    assert gt["ttype"] == 30 and gt["nbytes"] == 4096 * 256 * 2
                    gate_inp_loc[bid] = (base_path, old_dstart + gt["off"], gt["nbytes"])
                if bn in bmap:
                    bt = bmap[bn]
                    assert bt["ttype"] == 0 and bt["nbytes"] == 256 * 4
                    probs_b_loc[bid] = (base_path, old_dstart + bt["off"], bt["nbytes"])

        new_tensors = []
        active_moe_bid = None
        seen_moe_parts = set()

        for t in base_tensors:
            nm = t["name"]
            m = re.match(r"blk\.(\d+)\.(.+)$", nm)
            if m:
                bid = int(m.group(1))
                suf = m.group(2)
                if suf in ("ffn_down_exps.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight"):
                    km = k_for_layer(bid)
                    gu_tt = 16 if bid in IQ2_GU_LAYERS else (29 if bid in IQ1M_GU_LAYERS else 19)
                    down_tt = 39 if bid in (41, 42) else 16
                    sf_moe = weight_map[f"layers.{bid}.ffn.experts.0.w1.weight"]

                    if suf == "ffn_down_exps.weight":
                        assert active_moe_bid is None, f"MoE layer {active_moe_bid} incomplete before {nm}"
                        active_moe_bid = bid
                        seen_moe_parts = {"down"}
                        new_tensors.append({
                            "name": nm,
                            "dims": [km, 4096, 256],
                            "new_tt": down_tt,
                            "mode": "moe_down_trigger",
                            "bid": bid,
                            "km": km,
                            "gu_tt": gu_tt,
                            "down_tt": down_tt,
                            "sf_file": sf_moe,
                        })
                        if km < 2048:
                            new_tensors.append({
                                "name": f"blk.{bid}.ffn_down_exps.bias",
                                "dims": [4096, 256],
                                "new_tt": 0,
                                "mode": "moe_down_bias",
                                "bid": bid,
                            })
                    elif suf == "ffn_gate_exps.weight":
                        assert active_moe_bid == bid
                        seen_moe_parts.add("gate")
                        new_tensors.append({
                            "name": nm,
                            "dims": [4096, km, 256],
                            "new_tt": gu_tt,
                            "mode": "moe_gate",
                            "bid": bid,
                        })
                    elif suf == "ffn_up_exps.weight":
                        assert active_moe_bid == bid
                        seen_moe_parts.add("up")
                        new_tensors.append({
                            "name": nm,
                            "dims": [4096, km, 256],
                            "new_tt": gu_tt,
                            "mode": "moe_up",
                            "bid": bid,
                        })
                    if len(seen_moe_parts) == 3:
                        active_moe_bid = None
                        seen_moe_parts = set()
                    continue

                ntt = dense_target_ttype(bid, suf, t["ttype"])
                if ntt != t["ttype"] or (args.imatrix_2d and ntt in (12, 13, 14) and suf in LAYER_MAP):
                    if not args.imatrix_2d and nm in df_cache_2d and df_cache_2d[nm]["tt"] == ntt:
                        new_tensors.append({
                            "name": nm,
                            "dims": list(t["dims"]),
                            "new_tt": ntt,
                            "mode": "df_pack_copy",
                            "pack_off": int(df_cache_2d[nm]["off"]),
                        })
                    else:
                        st_key = f"layers.{bid}.{LAYER_MAP[suf]}"
                        new_tensors.append({
                            "name": nm,
                            "dims": list(t["dims"]),
                            "new_tt": ntt,
                            "mode": "quant_2d",
                            "st_key": st_key,
                            "sf_file": weight_map[st_key],
                        })
                    continue

            if nm == "output.weight" and args.imatrix_2d and t["ttype"] in (12, 13, 14):
                st_key = ROOT_MAP[nm]
                new_tensors.append({
                    "name": nm,
                    "dims": list(t["dims"]),
                    "new_tt": t["ttype"],
                    "mode": "quant_2d",
                    "st_key": st_key,
                    "sf_file": weight_map[st_key],
                })
                continue

            assert t["ttype"] != 42, f"Unupgraded Fractal-64 tensor {nm}"
            new_tensors.append({
                "name": nm,
                "dims": list(t["dims"]),
                "new_tt": t["ttype"],
                "mode": "local_copy",
                "old_abs_off": old_dstart + t["off"],
            })

        assert active_moe_bid is None
        total_split_tensors += len(new_tensors)
        raw_shards.append((s_idx, base_path, kv_prefix, new_tensors))

    assert total_split_tensors == 1362, f"Expected 1,362 total split tensors, got {total_split_tensors}"

    s1_out = f"{args.out_prefix}-00001-of-00003.gguf"
    if args.dry_run:
        s1_sz = os.path.getsize(f"{args.base_gguf_prefix}-00001-of-00003.gguf")
    else:
        s1_sz = rewrite_shard1(f"{args.base_gguf_prefix}-00001-of-00003.gguf", s1_out, total_split_tensors)
        print(f"[Shard 00001] Wrote {s1_out} ({s1_sz:,} B, split.tensors.count={total_split_tensors})", flush=True)

    shard_plans = []
    sf_refcnt = {}
    sf_order = []
    n_q2d = 0
    n_c2d = 0
    n_moe = 0
    n_reused_gu = 0

    for s_idx, base_path, kv_prefix, new_tensors in raw_shards:
        hdr_out = rewrite_kv_prefix(kv_prefix, new_nt=len(new_tensors), total_split_tensors=total_split_tensors)
        cur_off = 0
        for idx, t in enumerate(new_tensors):
            ncol = t["dims"][0]
            nrow = 1
            for d in t["dims"][1:]:
                nrow *= d
            nb = nrow * ROW_BYTES[t["new_tt"]](ncol)
            if idx > 0:
                cur_off = (cur_off + 31) // 32 * 32
            noff = cur_off
            cur_off += nb
            t["ncol"] = ncol
            t["nrow"] = nrow
            t["new_off"] = noff
            t["new_nbytes"] = nb

            hdr_out.extend(encode_gguf_str(t["name"]))
            hdr_out.extend(struct.pack("<I", len(t["dims"])))
            hdr_out.extend(struct.pack(f"<{len(t['dims'])}Q", *t["dims"]))
            hdr_out.extend(struct.pack("<IQ", t["new_tt"], noff))

        pad = ((len(hdr_out) + 31) // 32 * 32) - len(hdr_out)
        hdr_out.extend(b"\x00" * pad)
        hdr_bytes = bytes(hdr_out)
        new_dstart = len(hdr_bytes)
        expected_fsize = new_dstart + cur_off

        out_path = f"{args.out_prefix}-{s_idx}-of-00003.gguf"
        part_path = out_path + ".part"
        shard_done = False
        resume_idx = 0
        resume_written = 0

        if not args.dry_run:
            if os.path.isfile(out_path) and os.path.getsize(out_path) == expected_fsize:
                shard_done = True
                resume_idx = len(new_tensors)
                resume_written = expected_fsize
            elif os.path.isfile(part_path):
                part_sz = os.path.getsize(part_path)
                if part_sz >= new_dstart:
                    with open(part_path, "rb") as pf:
                        disk_hdr = pf.read(new_dstart)
                    if disk_hdr == hdr_bytes:
                        resume_written = new_dstart
                        in_moe_grp = False
                        for idx, t in enumerate(new_tensors):
                            if t["mode"] == "moe_down_trigger":
                                in_moe_grp = True
                            elif t["mode"] == "moe_up":
                                in_moe_grp = False
                            end_pos = new_dstart + t["new_off"] + t["new_nbytes"]
                            if end_pos > part_sz:
                                break
                            if not in_moe_grp:
                                resume_idx = idx + 1
                                resume_written = end_pos

        if not shard_done:
            for t in new_tensors[resume_idx:]:
                sf = t.get("sf_file")
                if sf is not None:
                    if sf not in sf_refcnt:
                        sf_refcnt[sf] = 0
                        sf_order.append(sf)
                    sf_refcnt[sf] += 1
                if t["mode"] == "quant_2d":
                    n_q2d += 1
                elif t["mode"] == "df_pack_copy":
                    n_c2d += 1
                elif t["mode"] == "moe_down_trigger":
                    n_moe += 1
                    bid = t["bid"]
                    gtt = t["gu_tt"]
                    if (bid, "gate", gtt) in base_preq_gu and (bid, "up", gtt) in base_preq_gu:
                        n_reused_gu += 1

        shard_plans.append((s_idx, base_path, hdr_bytes, new_dstart, expected_fsize, new_tensors, shard_done, resume_idx, resume_written))

    total_planned = s1_sz + sum(sp[4] for sp in shard_plans)
    cap_bytes = 58 * (1024 ** 3)
    print(
        f"[Plan] MaxQ-v4 exact size = {total_planned:,} B ({total_planned / (1024**3):.6f} GiB, "
        f"margin below 58.000 GiB = {(cap_bytes - total_planned) / (1024**2):.2f} MiB)\n"
        f"       split.tensors.count = {total_split_tensors} | 2D fresh={n_q2d}, cached={n_c2d} | "
        f"MoE layers = {n_moe} (reusing {n_reused_gu} donor-free pre-quantized gate/up layers) | "
        f"{len(sf_order)} upstream 0731 safetensor shards queued",
        flush=True,
    )
    assert total_planned < cap_bytes, f"FATAL: {total_planned} >= 58.000 GiB ({cap_bytes})!"
    if args.dry_run:
        return

    print(f"[Init] Loading @antirez 1.5M routed-MoE imatrix from {args.imatrix} ...", flush=True)
    im = load_imatrix_ctypes(args.imatrix, lib)
    assert len(im) == 129, f"Expected 129 imatrix entries, got {len(im)}"
    im2d = load_imatrix_ctypes(args.imatrix_2d, lib) if (args.imatrix_2d and os.path.isfile(args.imatrix_2d)) else {}
    if im2d:
        print(f"[Init] Loaded {len(im2d)} donor-free 0731 2D imatrix entries from {args.imatrix_2d}", flush=True)

    cond = threading.Condition()
    ready_set = set()
    active_on_disk = set()
    prefetch_err = []

    for fn in os.listdir(args.shards_dir):
        fp = os.path.join(args.shards_dir, fn)
        if fn in sf_refcnt and safetensor_is_complete(fp):
            ready_set.add(fn)
            active_on_disk.add(fn)
        else:
            try:
                os.unlink(fp)
            except OSError:
                pass
    if ready_set:
        print(f"[Cache] Reusing {len(ready_set)} pre-validated 0731 shards in {args.shards_dir}: {sorted(ready_set)}", flush=True)

    dl_pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.dl_workers)

    def prefetch_worker():
        try:
            for sf_file in sf_order:
                with cond:
                    if sf_file in ready_set:
                        continue
                    while len(active_on_disk) >= args.max_buffered_shards:
                        cond.wait(timeout=1.0)
                t_dl = time.time()
                dst_p = os.path.join(args.shards_dir, sf_file)
                download_safetensor_parallel(dl_pool, f"{args.hf_base}/{sf_file}", dst_p)
                sz_gib = os.path.getsize(dst_p) / (1024 ** 3)
                with cond:
                    ready_set.add(sf_file)
                    active_on_disk.add(sf_file)
                    cond.notify_all()
                print(
                    f"  [Prefetch] {sf_file} ({sz_gib:.2f} GiB) in {time.time()-t_dl:.1f}s "
                    f"(buffered={len(active_on_disk)}/{args.max_buffered_shards})",
                    flush=True,
                )
        except Exception as exc:
            with cond:
                prefetch_err.append(exc)
                cond.notify_all()

    pf_thread = threading.Thread(target=prefetch_worker, daemon=True)
    pf_thread.start()

    open_readers = {}

    def acquire_reader(sf_file: str) -> SafetensorReader:
        if sf_file not in open_readers:
            with cond:
                while sf_file not in ready_set and not prefetch_err:
                    cond.wait(timeout=1.0)
                if prefetch_err:
                    raise RuntimeError(f"Safetensor prefetch failed: {prefetch_err[0]}") from prefetch_err[0]
            open_readers[sf_file] = SafetensorReader(os.path.join(args.shards_dir, sf_file))
        return open_readers[sf_file]

    def release_ref(sf_file: str) -> None:
        sf_refcnt[sf_file] -= 1
        if sf_refcnt[sf_file] == 0:
            rd = open_readers.pop(sf_file)
            rd.close()
            try:
                os.unlink(os.path.join(args.shards_dir, sf_file))
            except OSError:
                pass
            with cond:
                active_on_disk.discard(sf_file)
                cond.notify_all()

    copy_buf = bytearray(64 * 1024 * 1024)
    q2d_max  = 512 * 1024 * 1024
    q2d_ba   = bytearray(q2d_max)
    q2d_c    = (ctypes.c_uint8 * q2d_max).from_buffer(q2d_ba)
    q2d_mv   = memoryview(q2d_ba)

    gu_max   = 256 * 2048 * 16 * 66
    dn_max   = 256 * 4096 * 64 * 17
    bi_bytes = 256 * 4096 * 4

    moe_g_ba = bytearray(gu_max)
    moe_u_ba = bytearray(gu_max)
    moe_d_ba = bytearray(dn_max)
    moe_b_ba = bytearray(bi_bytes)

    moe_g_c  = (ctypes.c_uint8 * gu_max).from_buffer(moe_g_ba)
    moe_u_c  = (ctypes.c_uint8 * gu_max).from_buffer(moe_u_ba)
    moe_d_c  = (ctypes.c_uint8 * dn_max).from_buffer(moe_d_ba)
    moe_b_c  = (ctypes.c_float * (256 * 4096)).from_buffer(moe_b_ba)

    moe_g_mv = memoryview(moe_g_ba)
    moe_u_mv = memoryview(moe_u_ba)
    moe_d_mv = memoryview(moe_d_ba)
    moe_b_mv = memoryview(moe_b_ba)

    loc_gu_max = 256 * 2048 * 16 * 66
    loc_gq_ba = bytearray(loc_gu_max)
    loc_uq_ba = bytearray(loc_gu_max)
    loc_gq_mv = memoryview(loc_gq_ba)
    loc_uq_mv = memoryview(loc_uq_ba)
    loc_gq_c  = (ctypes.c_uint8 * loc_gu_max).from_buffer(loc_gq_ba)
    loc_uq_c  = (ctypes.c_uint8 * loc_gu_max).from_buffer(loc_uq_ba)

    cal_bf_ba = bytearray(exp_bf16_sz)
    cal_id_ba = bytearray(exp_ids_sz)
    cal_bf_mv = memoryview(cal_bf_ba)
    cal_id_mv = memoryview(cal_id_ba)
    cal_bf_c  = (ctypes.c_uint16 * (args.n_cal_tokens * 4096)).from_buffer(cal_bf_ba)
    cal_id_c  = (ctypes.c_uint8 * exp_ids_sz).from_buffer(cal_id_ba)

    def read_file_into(path: str, dst_mv: memoryview, length: int) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            pos = 0
            while pos < length:
                got = os.preadv(fd, [dst_mv[pos:length]], pos)
                if got <= 0:
                    raise RuntimeError(f"Short read on {path}")
                pos += got
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except Exception:
                pass
        finally:
            os.close(fd)

    def read_local_slice(path: str, cur_path: str, cur_fd: int, abs_off: int, length: int, dst_mv: memoryview) -> None:
        def _fill(fd: int):
            pos = 0
            while pos < length:
                got = os.preadv(fd, [dst_mv[pos:length]], abs_off + pos)
                if got <= 0:
                    raise RuntimeError(f"Unexpected EOF in {path} at {abs_off + pos}")
                pos += got
        if path == cur_path:
            _fill(cur_fd)
        else:
            xfd = os.open(path, os.O_RDONLY)
            try:
                _fill(xfd)
            finally:
                os.close(xfd)

    pack_fd = os.open(df_pack_path, os.O_RDONLY) if df_pack_path else -1
    active_cache_bid = None
    done_t = 0

    try:
        for s_idx, base_path, hdr_bytes, new_dstart, expected_fsize, tensors, shard_done, resume_idx, resume_written in shard_plans:
            out_path = f"{args.out_prefix}-{s_idx}-of-00003.gguf"
            if shard_done:
                print(f"[Shard {s_idx}] Already complete ({expected_fsize:,} B), skipping!", flush=True)
                done_t += len(tensors)
                continue

            part_path = out_path + ".part"
            src_fd = os.open(base_path, os.O_RDONLY)
            t_sh = time.time()
            if resume_written > 0:
                dst_fd = os.open(part_path, os.O_WRONLY)
                os.ftruncate(dst_fd, resume_written)
                os.lseek(dst_fd, resume_written, os.SEEK_SET)
                written = resume_written
                done_t += resume_idx
                print(
                    f"\n[Shard {s_idx}] RESUMING {out_path} at tensor {resume_idx}/{len(tensors)} "
                    f"({written / (1024**3):.3f} / {expected_fsize / (1024**3):.3f} GiB)...",
                    flush=True,
                )
            else:
                dst_fd = os.open(part_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
                os.write(dst_fd, hdr_bytes)
                written = len(hdr_bytes)
                print(
                    f"\n[Shard {s_idx}] Building {out_path} ({len(tensors)} tensors, {expected_fsize / (1024**3):.3f} GiB)...",
                    flush=True,
                )

            for t in tensors[resume_idx:]:
                target_pos = new_dstart + t["new_off"]
                if written < target_pos:
                    os.write(dst_fd, b"\x00" * (target_pos - written))
                    written = target_pos
                assert written == target_pos, f"Offset mismatch on {t['name']}: {written} != {target_pos}"

                nb = t["new_nbytes"]
                mode = t["mode"]
                t0 = time.time()

                if mode == "local_copy":
                    if t["name"] == "output_norm.weight" and abs(args.output_norm_tau - 1.0) > 1e-7:
                        assert nb == 16384
                        raw_on = os.pread(src_fd, nb, t["old_abs_off"])
                        vals_on = struct.unpack("<4096f", raw_on)
                        os.write(dst_fd, struct.pack("<4096f", *[v * args.output_norm_tau for v in vals_on]))
                    else:
                        copy_local_range(src_fd, t["old_abs_off"], dst_fd, nb, copy_buf)
                    written += nb

                elif mode == "df_pack_copy":
                    assert pack_fd >= 0
                    copy_local_range(pack_fd, t["pack_off"], dst_fd, nb, copy_buf)
                    written += nb

                elif mode == "quant_2d":
                    sf = t["sf_file"]
                    st = acquire_reader(sf)
                    st_key = t["st_key"]
                    dtype, _shape, w_ptr, _ = st.get_info(st_key)
                    s_key = st_key.removesuffix(".weight") + ".scale" if st_key.endswith(".weight") else ""
                    s_ptr = st.get_info(s_key)[2] if (s_key and s_key in st.hdr) else 0
                    assert nb <= q2d_max
                    im_arr = im2d.get(t["name"])
                    im_ptr = im_arr if im_arr is not None else None
                    im_nval = len(im_arr) if im_arr is not None else 0
                    q2d_rc = lib.mq4_quantize_2d(w_ptr, DTYPE_CODE[dtype], s_ptr, q2d_c, t["new_tt"], t["nrow"], t["ncol"], im_ptr, im_nval)
                    if im_ptr is not None and t["new_tt"] != 8 and q2d_rc != 1:
                        raise RuntimeError(f"mq4_quantize_2d failed imatrix validation on {t['name']} (rc={q2d_rc})")
                    write_all(dst_fd, q2d_mv, nb)
                    written += nb
                    release_ref(sf)

                elif mode == "moe_down_trigger":
                    bid = t["bid"]
                    km = t["km"]
                    gu_tt = t["gu_tt"]
                    down_tt = t["down_tt"]
                    sf = t["sf_file"]
                    st = acquire_reader(sf)

                    wg_ptrs = (ctypes.c_void_p * 256)()
                    sg_ptrs = (ctypes.c_void_p * 256)()
                    wu_ptrs = (ctypes.c_void_p * 256)()
                    su_ptrs = (ctypes.c_void_p * 256)()
                    wd_ptrs = (ctypes.c_void_p * 256)()
                    sd_ptrs = (ctypes.c_void_p * 256)()
                    for e in range(256):
                        wg_ptrs[e] = st.get_info(f"layers.{bid}.ffn.experts.{e}.w1.weight")[2]
                        sg_ptrs[e] = st.get_info(f"layers.{bid}.ffn.experts.{e}.w1.scale")[2]
                        wd_ptrs[e] = st.get_info(f"layers.{bid}.ffn.experts.{e}.w2.weight")[2]
                        sd_ptrs[e] = st.get_info(f"layers.{bid}.ffn.experts.{e}.w2.scale")[2]
                        wu_ptrs[e] = st.get_info(f"layers.{bid}.ffn.experts.{e}.w3.weight")[2]
                        su_ptrs[e] = st.get_info(f"layers.{bid}.ffn.experts.{e}.w3.scale")[2]

                    if (bid, "gate", gu_tt) in base_preq_gu and (bid, "up", gu_tt) in base_preq_gu:
                        gp, goff, gnb = base_preq_gu[(bid, "gate", gu_tt)]
                        up, uoff, unb = base_preq_gu[(bid, "up", gu_tt)]
                        read_local_slice(gp, base_path, src_fd, goff, gnb, loc_gq_mv)
                        read_local_slice(up, base_path, src_fd, uoff, unb, loc_uq_mv)
                        raw_gq_ptr = loc_gq_c
                        raw_uq_ptr = loc_uq_c
                        gu_src_tag = "cached-df"
                    else:
                        raw_gq_ptr = None
                        raw_uq_ptr = None
                        gu_src_tag = "fresh-0731"

                    read_file_into(os.path.join(args.calib_dir, f"dsv4_calib_L{bid:02d}.bf16"), cal_bf_mv, exp_bf16_sz)
                    read_file_into(os.path.join(args.calib_dir, f"dsv4_ids_L{bid:02d}.u8"),   cal_id_mv, exp_ids_sz)

                    if bid >= 3:
                        rp, roff, rnb = gate_inp_loc[bid]
                        bp, boff, bnb = probs_b_loc[bid]
                        r_ba = bytearray(rnb)
                        b_ba = bytearray(bnb)
                        read_local_slice(rp, base_path, src_fd, roff, rnb, memoryview(r_ba))
                        read_local_slice(bp, base_path, src_fd, boff, bnb, memoryview(b_ba))
                        r_c = (ctypes.c_uint8 * rnb).from_buffer(r_ba)
                        b_c = (ctypes.c_uint8 * bnb).from_buffer(b_ba)
                    else:
                        r_c = None
                        b_c = None

                    im_g = im[f"blk.{bid}.ffn_gate_exps.weight"]
                    im_u = im[f"blk.{bid}.ffn_up_exps.weight"]
                    im_d = im[f"blk.{bid}.ffn_down_exps.weight"]
                    mean_alpha = ctypes.c_float(0.0)

                    retained = lib.mq4_collapse_moe_layer(
                        wg_ptrs, sg_ptrs, wu_ptrs, su_ptrs, wd_ptrs, sd_ptrs,
                        raw_gq_ptr, raw_uq_ptr,
                        im_g, im_u, im_d,
                        cal_bf_c, cal_id_c, args.n_cal_tokens,
                        r_c, b_c,
                        gu_tt, down_tt, km,
                        moe_g_c, moe_u_c, moe_d_c, moe_b_c,
                        ctypes.byref(mean_alpha),
                    )
                    release_ref(sf)

                    write_all(dst_fd, moe_d_mv, nb)
                    written += nb
                    active_cache_bid = bid
                    el = time.time() - t_start
                    print(
                        f"  [{done_t+1:4d}/{total_split_tensors} | {el/60:5.1f}m] MoE L{bid:02d} "
                        f"(K={km:4d}, gu=t{gu_tt}:{gu_src_tag}, dn=t{down_tt}) -> "
                        f"retained={retained:6.2f}%, alpha={mean_alpha.value:6.4f} in {time.time()-t0:.2f}s",
                        flush=True,
                    )

                elif mode == "moe_down_bias":
                    assert active_cache_bid == t["bid"] and nb == bi_bytes
                    write_all(dst_fd, moe_b_mv, nb)
                    written += nb

                elif mode == "moe_gate":
                    assert active_cache_bid == t["bid"]
                    write_all(dst_fd, moe_g_mv, nb)
                    written += nb

                elif mode == "moe_up":
                    assert active_cache_bid == t["bid"]
                    write_all(dst_fd, moe_u_mv, nb)
                    written += nb
                    active_cache_bid = None

                else:
                    raise RuntimeError(f"Unknown mode {mode}")

                try:
                    os.posix_fadvise(dst_fd, (target_pos // 4096) * 4096, (nb // 4096) * 4096, os.POSIX_FADV_DONTNEED)
                except Exception:
                    pass
                done_t += 1

            assert written == expected_fsize, f"Shard {s_idx} size mismatch: {written} != {expected_fsize}"
            os.fsync(dst_fd)
            os.close(dst_fd)
            os.close(src_fd)
            os.replace(part_path, out_path)
            print(f"[Shard {s_idx}] COMPLETED {out_path} ({written:,} B) in {(time.time()-t_sh)/60:.1f}m!", flush=True)
    finally:
        if pack_fd >= 0:
            os.close(pack_fd)

    pf_thread.join(timeout=5.0)
    dl_pool.shutdown(wait=False)
    try:
        os.rmdir(args.shards_dir)
    except OSError:
        pass
    print(
        f"\n[SUCCESS] MaxQ-v4 COMPLETE: {total_planned:,} B ({total_planned / (1024**3):.6f} GiB) "
        f"in {(time.time()-t_start)/60:.1f} minutes!",
        flush=True,
    )


if __name__ == "__main__":
    main()
