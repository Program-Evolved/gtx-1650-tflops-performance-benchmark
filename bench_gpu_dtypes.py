#!/usr/bin/env python3
"""Benchmark FP32 vs FP16 vs BF16 vs TF32 on any CUDA GPU.

Two levels:

1. Raw GEMM: square matmuls in each dtype, measured with CUDA events.
2. Realistic GPT training step: forward + backward + AdamW for a small
   LLaMA-style transformer, with optional gradient checkpointing.

The point is to answer, on *your* GPU/driver/PyTorch build:

    "Is FP16 autocast actually faster than FP32 here?"

It is not always. On a GTX 1650 + PyTorch 2.10/cu128, for example, FP32 is
about 4x faster than FP16 autocast for this workload, because the FP16 GEMM
path is slow on that device/build. This script measures instead of assuming.

Examples
--------
    python bench_gpu_dtypes.py
    python bench_gpu_dtypes.py --quick
    python bench_gpu_dtypes.py --batch-size 16 --seq-len 512
    python bench_gpu_dtypes.py --json results.json
    python bench_gpu_dtypes.py --sizes 1024,2048,4096 --gemm-iters 20

Only PyTorch is required. The model benchmark uses random token IDs, so no
dataset or tokenizer is needed.

Notes
-----
* FP16 autocast is the normal mixed-precision path: FP32 master weights,
  FP16 matmuls, GradScaler.
* FP16 native (optional) casts the whole model to half. It is included because
  raw FP16 GEMM speed and autocast speed can differ; it uses no GradScaler
  (native half gradients cannot be unscaled by GradScaler).
* BF16 is included when the build reports support. On pre-Ampere NVIDIA GPUs
  this may be emulated rather than accelerated.
* TF32 is included only when the GPU is Ampere or newer (compute >= 8.0).
* The script auto-reduces the model batch size if FP32 does not fit, so it
  also works on small GPUs. Use --skip-model or --skip-gemm for a shorter run.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _version_tuple(v: str) -> Tuple[int, ...]:
    parts: List[int] = []
    for piece in v.split("+")[0].split("."):
        digits = ""
        for ch in piece:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


_TORCH_VER = _version_tuple(torch.__version__)
SDPA_HAS_GQA = _TORCH_VER >= (2, 5)


def set_fp32_precision(mode: str) -> None:
    """'highest' = true FP32; 'high' = TF32 on Ampere+."""
    try:
        torch.set_float32_matmul_precision(mode)
    except Exception:
        pass
    try:
        allow = mode != "highest"
        torch.backends.cuda.matmul.allow_tf32 = allow
        torch.backends.cudnn.allow_tf32 = allow
    except Exception:
        pass


def fmt_mem(gb: float) -> str:
    return f"{gb:.2f} GB"


def fmt_num(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "-"
    if isinstance(x, float) and not math.isfinite(x):
        return "inf" if x > 0 else "nan"
    return f"{x:,.{digits}f}"


# ---------------------------------------------------------------------------
# GPU information
# ---------------------------------------------------------------------------

def collect_gpu_info() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU found. This script benchmarks CUDA devices.\n"
            "Install a CUDA build of PyTorch and run it on the GPU machine."
        )

    props = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    is_rocm = getattr(torch.version, "hip", None) is not None

    try:
        bf16_supported = bool(torch.cuda.is_bf16_supported())
    except Exception:
        bf16_supported = False

    tf32_supported = (not is_rocm) and capability >= (8, 0)
    bf16_hardware = (not is_rocm) and capability >= (8, 0)

    driver: Optional[str] = None
    if not is_rocm:
        try:
            driver = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).strip().splitlines()[0]
        except Exception:
            driver = None

    return {
        "name": torch.cuda.get_device_name(0),
        "capability": capability,
        "capability_str": f"{capability[0]}.{capability[1]}",
        "total_memory_gb": props.total_memory / 1e9,
        "multi_processor_count": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hip": getattr(torch.version, "hip", None),
        "driver": driver,
        "bf16_supported": bf16_supported,
        "bf16_hardware": bf16_hardware,
        "tf32_supported": tf32_supported,
        "is_rocm": is_rocm,
        "fp16_reduced_precision_reduction": bool(
            getattr(
                torch.backends.cuda.matmul,
                "allow_fp16_reduced_precision_reduction",
                False,
            )
        ),
    }


def print_gpu_info(info: Dict[str, Any]) -> None:
    print("=" * 88)
    print("GPU dtype benchmark")
    print("=" * 88)
    print(f"GPU:                 {info['name']}")
    print(f"Compute capability:  {info['capability_str']}")
    print(f"VRAM:                {fmt_mem(info['total_memory_gb'])}")
    print(f"SMs:                 {info['multi_processor_count']}")
    print(f"PyTorch:             {info['torch']}")
    print(f"CUDA runtime:        {info['cuda']}")
    if info["hip"]:
        print(f"HIP runtime:         {info['hip']}")
    if info["driver"]:
        print(f"Driver:              {info['driver']}")
    print(
        "BF16 supported:      "
        f"{info['bf16_supported']}"
        + (
            " (hardware)"
            if info["bf16_hardware"]
            else " (may be emulated on this GPU)"
        )
    )
    print(f"TF32 supported:      {info['tf32_supported']}")
    print(
        "FP16 reduced-prec:   "
        f"{info['fp16_reduced_precision_reduction']}"
    )
    print("=" * 88)


# ---------------------------------------------------------------------------
# Raw GEMM benchmark
# ---------------------------------------------------------------------------

def gemm_accuracy(dtype: torch.dtype, mode: str, n: int = 512) -> Optional[float]:
    """Relative Frobenius error versus an FP32-highest reference."""
    try:
        torch.cuda.empty_cache()
        a = torch.randn(n, n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, n, device="cuda", dtype=torch.float32)

        set_fp32_precision("highest")
        ref = a @ b

        if mode == "tf32":
            set_fp32_precision("high")
            out = (a @ b).float()
        elif dtype == torch.float32:
            out = ref
        else:
            out = (a.to(dtype) @ b.to(dtype)).float()

        denom = ref.norm().item()
        rel = ((out - ref).norm().item() / denom) if denom > 0 else float("nan")
        del a, b, ref, out
        torch.cuda.empty_cache()
        return rel
    except Exception:
        return None


def bench_gemm(
    n: int,
    dtype: torch.dtype,
    mode: str,
    iters: int,
    warmup: int,
) -> Tuple[float, float]:
    set_fp32_precision("high" if mode == "tf32" else "highest")
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)

    for _ in range(max(1, warmup)):
        c = a @ b
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(max(1, iters)):
        c = a @ b
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / max(1, iters)
    tflops = (2.0 * n**3) / (ms * 1e-3) / 1e12

    del a, b, c
    torch.cuda.empty_cache()
    set_fp32_precision("highest")
    return ms, tflops


def run_gemm(
    info: Dict[str, Any],
    sizes: List[int],
    iters: int,
    warmup: int,
) -> List[Dict[str, Any]]:
    modes: List[Tuple[str, torch.dtype, str]] = [
        ("fp32", torch.float32, "FP32"),
        ("fp16", torch.float16, "FP16"),
    ]
    if info["bf16_supported"]:
        modes.append(("bf16", torch.bfloat16, "BF16"))
    if info["tf32_supported"]:
        modes.append(("tf32", torch.float32, "TF32"))

    accuracy: Dict[str, Optional[float]] = {}
    for mode, dtype, _ in modes:
        accuracy[mode] = gemm_accuracy(dtype, mode)

    results: List[Dict[str, Any]] = []

    for n in sizes:
        free, _ = torch.cuda.mem_get_info()
        # a, b, c + a little slack; FP32 is the worst case.
        need = 4.0 * n * n * 4.0
        if need > free * 0.80:
            print(
                f"  skip GEMM n={n}: need ~{need/1e9:.2f} GB, "
                f"free {free/1e9:.2f} GB"
            )
            continue

        for mode, dtype, label in modes:
            try:
                ms, tflops = bench_gemm(
                    n, dtype, mode, iters=iters, warmup=warmup
                )
                results.append(
                    {
                        "mode": mode,
                        "label": label,
                        "n": n,
                        "ms": ms,
                        "tflops": tflops,
                        "relerr": accuracy.get(mode),
                    }
                )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                results.append(
                    {
                        "mode": mode,
                        "label": label,
                        "n": n,
                        "error": f"OOM: {exc}",
                    }
                )
            except Exception as exc:
                torch.cuda.empty_cache()
                results.append(
                    {
                        "mode": mode,
                        "label": label,
                        "n": n,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    return results


def print_gemm_table(
    results: List[Dict[str, Any]], sizes: List[int]
) -> None:
    print("\n=== raw GEMM (CUDA events) ===")
    print(
        f"{'dtype':<8}{'n':>8}{'ms':>12}{'TFLOPS':>12}"
        f"{'vs FP32':>10}{'rel.err':>12}"
    )
    print("-" * 62)

    for n in sizes:
        rows = [r for r in results if r["n"] == n]
        if not rows:
            continue
        base = next(
            (
                r["tflops"]
                for r in rows
                if r["mode"] == "fp32" and "tflops" in r
            ),
            None,
        )
        for r in rows:
            if "error" in r:
                print(
                    f"{r['label']:<8}{n:>8}   ERROR: {r['error']}"
                )
                continue
            speedup = (
                (r["tflops"] / base)
                if base and base > 0
                else None
            )
            rel = r.get("relerr")
            rel_s = "-" if rel is None else f"{rel:.1e}"
            print(
                f"{r['label']:<8}{n:>8}{r['ms']:>12.2f}"
                f"{r['tflops']:>12.3f}"
                f"{(f'{speedup:.2f}x' if speedup is not None else '-'):>10}"
                f"{rel_s:>12}"
            )
        print("-" * 62)


# ---------------------------------------------------------------------------
# A small LLaMA-style GPT, for a realistic training step
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 16384
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int = 2
    n_embd: int = 512
    ffn_hidden: int = 1408
    grad_checkpointing: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        ).to(x.dtype)


def precompute_rope(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dim, 2, device=device).float()
            / head_dim
        )
    )
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    T = x.size(-2)
    cos = cos[:T].to(x.dtype)[None, None, :, :]
    sin = sin[:T].to(x.dtype)[None, None, :, :]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack(
        (x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1
    ).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head

        self.q_proj = nn.Linear(
            cfg.n_embd, cfg.n_head * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            cfg.n_head * self.head_dim, cfg.n_embd, bias=False
        )

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q = (
            self.q_proj(x)
            .view(B, T, self.n_head, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(B, T, self.n_kv_head, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(B, T, self.n_kv_head, self.head_dim)
            .transpose(1, 2)
        )

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if SDPA_HAS_GQA and self.n_kv_head != self.n_head:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=True
            )
        else:
            if self.n_kv_head != self.n_head:
                rep = self.n_head // self.n_kv_head
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


def chunked_ce(
    head: nn.Linear,
    h: torch.Tensor,
    targets: torch.Tensor,
    chunk: int = 64,
) -> torch.Tensor:
    """Cross-entropy over the LM head without materializing B*T*V at once.

    The caller supplies already-shifted targets: h[:, t] predicts targets[:, t].
    """
    B, T, D = h.shape
    total = h.new_zeros((), dtype=torch.float32)
    count = (targets != -100).sum().clamp_min(1)
    for t in range(0, T, chunk):
        end = min(t + chunk, T)
        logits = head(h[:, t:end])
        tgt = targets[:, t:end]
        total = total + F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
    return total / count


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(
            [Block(cfg) for _ in range(cfg.n_layer)]
        )
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.register_buffer(
            "rope_cos", torch.empty(0), persistent=False
        )
        self.register_buffer(
            "rope_sin", torch.empty(0), persistent=False
        )
        self._build_rope(cfg.block_size)

        self.apply(self._init_weights)
        for blk in self.blocks:
            nn.init.zeros_(blk.attn.o_proj.weight)
            nn.init.zeros_(blk.mlp.down.weight)

        self.head.weight = self.tok_emb.weight

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _build_rope(self, T: int) -> None:
        cos, sin = precompute_rope(
            self.cfg.n_embd // self.cfg.n_head,
            T,
            device=self.tok_emb.weight.device,
        )
        self.rope_cos = cos
        self.rope_sin = sin

    def forward_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        if T > self.rope_cos.size(0):
            self._build_rope(T)
        x = self.tok_emb(idx)
        for blk in self.blocks:
            if self.training and self.cfg.grad_checkpointing:
                x = checkpoint(
                    blk,
                    x,
                    self.rope_cos,
                    self.rope_sin,
                    use_reentrant=False,
                )
            else:
                x = blk(x, self.rope_cos, self.rope_sin)
        return self.ln_f(x)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor,
        chunk: int = 64,
    ) -> torch.Tensor:
        h = self.forward_hidden(idx)
        return chunked_ce(self.head, h, targets, chunk)


# ---------------------------------------------------------------------------
# Model training-step benchmark
# ---------------------------------------------------------------------------

def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    try:
        return torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=True,
        )
    except Exception:
        return torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )


def make_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    autocast_dtype: Optional[torch.dtype],
    x: torch.Tensor,
    y: torch.Tensor,
    chunk: int,
):
    """Return a closure that does one forward/backward/optimizer step."""

    def step() -> torch.Tensor:
        if autocast_dtype is not None:
            with torch.amp.autocast(
                "cuda", dtype=autocast_dtype, enabled=True
            ):
                loss = model(x, y, chunk=chunk)
        else:
            loss = model(x, y, chunk=chunk)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        return loss.detach()

    return step


def find_fitting_batch(
    cfg: ModelConfig,
    requested_batch: int,
    ce_chunk: int,
) -> int:
    """Use FP32 (the largest-memory mode) to find a batch that fits."""
    batch = max(1, requested_batch)
    while batch >= 1:
        model = None
        optimizer = None
        try:
            torch.cuda.empty_cache()
            gc.collect()
            model = GPT(cfg).cuda()
            optimizer = make_optimizer(model)
            x = torch.randint(
                0,
                cfg.vocab_size,
                (batch, cfg.block_size),
                device="cuda",
            )
            y = torch.cat([x[:, 1:], x[:, :1]], dim=1)
            loss = model(x, y, chunk=ce_chunk)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            del model, optimizer, x, y, loss
            torch.cuda.empty_cache()
            return batch
        except torch.cuda.OutOfMemoryError:
            if model is not None:
                del model
            if optimizer is not None:
                del optimizer
            torch.cuda.empty_cache()
            gc.collect()
            if batch == 1:
                return 0
            batch //= 2
        except Exception:
            if model is not None:
                del model
            if optimizer is not None:
                del optimizer
            torch.cuda.empty_cache()
            raise
    return 0


def run_model_mode(
    mode: str,
    cfg: ModelConfig,
    batch: int,
    warmup: int,
    steps: int,
    ce_chunk: int,
    compile_model: bool,
) -> Dict[str, Any]:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    set_fp32_precision(
        "high" if mode == "tf32" else "highest"
    )

    model: Optional[nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    scaler: Optional[torch.amp.GradScaler] = None

    try:
        model = GPT(cfg).cuda()
        if mode == "fp16_native":
            model = model.half()
        elif mode == "bf16_native":
            model = model.bfloat16()

        if compile_model:
            try:
                model = torch.compile(model)
            except Exception:
                pass

        optimizer = make_optimizer(model)
        scaler = torch.amp.GradScaler(
            "cuda", enabled=(mode == "fp16_autocast")
        )
        autocast_dtype: Optional[torch.dtype] = {
            "fp16_autocast": torch.float16,
            "bf16_autocast": torch.bfloat16,
        }.get(mode)

        x = torch.randint(
            0,
            cfg.vocab_size,
            (batch, cfg.block_size),
            device="cuda",
        )
        y = torch.cat([x[:, 1:], x[:, :1]], dim=1)

        step = make_step(
            model,
            optimizer,
            scaler,
            autocast_dtype,
            x,
            y,
            ce_chunk,
        )

        # Warmup (also compiles if torch.compile is enabled).
        t_warm = time.perf_counter()
        warm_loss: Optional[torch.Tensor] = None
        for _ in range(max(0, warmup)):
            warm_loss = step()
            warm_loss.item()
        torch.cuda.synchronize()
        warm_seconds = time.perf_counter() - t_warm

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss: Optional[torch.Tensor] = None
        for _ in range(max(1, steps)):
            loss = step()
        end.record()
        torch.cuda.synchronize()

        elapsed = start.elapsed_time(end) / 1000.0
        elapsed = max(elapsed, 1e-9)
        loss_value = float(loss.item()) if loss is not None else float("nan")
        finite = math.isfinite(loss_value)
        # A random-init model should be around ln(vocab_size). A much
        # larger loss means the mode went unstable (usually native FP16
        # without FP32 master weights), so do not recommend it.
        loss_limit = 2.0 * math.log(max(2, cfg.vocab_size)) + 5.0
        valid = finite and loss_value <= loss_limit
        tokens_per_s = batch * cfg.block_size * max(1, steps) / elapsed

        return {
            "mode": mode,
            "batch": batch,
            "seq_len": cfg.block_size,
            "tokens_per_step": batch * cfg.block_size,
            "steps": steps,
            "seconds": elapsed,
            "tokens_per_s": tokens_per_s,
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
            "loss": loss_value if finite else None,
            "valid": valid,
            "finite": finite,
            "warmup_seconds": warm_seconds,
        }
    except torch.cuda.OutOfMemoryError as exc:
        return {
            "mode": mode,
            "error": f"OOM: {exc}",
        }
    except Exception as exc:
        return {
            "mode": mode,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
        if scaler is not None:
            del scaler
        torch.cuda.empty_cache()
        gc.collect()
        set_fp32_precision("highest")


def run_model(
    info: Dict[str, Any],
    cfg: ModelConfig,
    batch: int,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    modes: List[str] = ["fp32", "fp16_autocast"]
    if not args.skip_native:
        modes.append("fp16_native")
    if info["bf16_supported"]:
        modes.extend(["bf16_autocast"])
        if not args.skip_native:
            modes.append("bf16_native")
    if info["tf32_supported"]:
        modes.append("tf32")

    results: List[Dict[str, Any]] = []
    for mode in modes:
        print(f"\n-- model: {mode} --")
        result = run_model_mode(
            mode,
            cfg,
            batch,
            warmup=args.model_warmup,
            steps=args.steps,
            ce_chunk=args.ce_chunk,
            compile_model=args.compile,
        )
        results.append(result)
        if "error" in result:
            print(f"   {result['error']}")
        else:
            print(
                f"   {result['tokens_per_s']:,.0f} tok/s | "
                f"{result['peak_gb']:.2f} GB | "
                f"loss {fmt_num(result['loss'])}"
            )
    return results


def print_model_table(results: List[Dict[str, Any]]) -> None:
    print("\n=== GPT training step ===")
    print(
        f"{'mode':<18}{'tok/s':>12}{'vs FP32':>10}"
        f"{'peak GB':>10}{'loss':>10}{'status':>10}"
    )
    print("-" * 70)
    base = next(
        (
            r["tokens_per_s"]
            for r in results
            if r["mode"] == "fp32" and "tokens_per_s" in r
        ),
        None,
    )

    for r in results:
        if "error" in r:
            print(f"{r['mode']:<18}   ERROR: {r['error']}")
            continue
        speedup = (
            (r["tokens_per_s"] / base)
            if base and base > 0
            else None
        )
        if r.get("valid"):
            status = "ok"
        elif not r.get("finite", True):
            status = "nonfinite"
        else:
            status = "unstable"
        print(
            f"{r['mode']:<18}{r['tokens_per_s']:>12,.0f}"
            f"{(f'{speedup:.2f}x' if speedup is not None else '-'):>10}"
            f"{r['peak_gb']:>10.2f}"
            f"{fmt_num(r.get('loss'), 3):>10}"
            f"{status:>10}"
        )
    print("-" * 70)
    print(
        "nonfinite/unstable = NaN, inf, or a loss far above random init; "
        "throughput is shown but the mode is excluded from the "
        "recommendation.\n"
        "fp16_native is diagnostic (whole model in half, no FP32 master "
        "weights/GradScaler), not a recommended training setup."
    )


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def build_recommendation(
    info: Dict[str, Any],
    gemm: List[Dict[str, Any]],
    model: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []

    valid_model = [
        r
        for r in model
        if "error" not in r and r.get("valid") and r.get("tokens_per_s")
    ]
    fp32_model = next(
        (r for r in valid_model if r["mode"] == "fp32"), None
    )
    fp16_model = next(
        (r for r in valid_model if r["mode"] == "fp16_autocast"),
        None,
    )
    bf16_model = next(
        (r for r in valid_model if r["mode"] == "bf16_autocast"),
        None,
    )
    tf32_model = next(
        (r for r in valid_model if r["mode"] == "tf32"), None
    )

    if valid_model:
        best = max(valid_model, key=lambda r: r["tokens_per_s"])
        if fp32_model:
            speed = best["tokens_per_s"] / fp32_model["tokens_per_s"]
            lines.append(
                f"Fastest training mode: {best['mode']} "
                f"({best['tokens_per_s']:,.0f} tok/s, "
                f"{speed:.2f}x FP32)."
            )
        else:
            lines.append(
                f"Fastest training mode: {best['mode']} "
                f"({best['tokens_per_s']:,.0f} tok/s)."
            )

        if fp16_model and fp32_model:
            ratio = (
                fp16_model["tokens_per_s"]
                / fp32_model["tokens_per_s"]
            )
            if ratio < 0.95:
                lines.append(
                    "FP16 autocast is slower than FP32 for this "
                    f"workload ({ratio:.2f}x FP32)."
                )
            elif ratio > 1.05:
                lines.append(
                    "FP16 autocast is faster than FP32 for this "
                    f"workload ({ratio:.2f}x FP32)."
                )
            else:
                lines.append(
                    "FP16 autocast and FP32 are roughly equal "
                    f"({ratio:.2f}x FP32)."
                )

        if bf16_model and fp32_model:
            ratio = (
                bf16_model["tokens_per_s"]
                / fp32_model["tokens_per_s"]
            )
            lines.append(
                f"BF16 autocast: {ratio:.2f}x FP32."
            )

        if tf32_model and fp32_model:
            ratio = (
                tf32_model["tokens_per_s"]
                / fp32_model["tokens_per_s"]
            )
            lines.append(f"TF32: {ratio:.2f}x FP32.")

    # Raw GEMM verdict at the largest measured size.
    if gemm:
        max_n = max(r["n"] for r in gemm if "tflops" in r) if any(
            "tflops" in r for r in gemm
        ) else None
        if max_n is not None:
            rows = [
                r
                for r in gemm
                if r["n"] == max_n and "tflops" in r
            ]
            base = next(
                (
                    r["tflops"]
                    for r in rows
                    if r["mode"] == "fp32"
                ),
                None,
            )
            fp16 = next(
                (
                    r["tflops"]
                    for r in rows
                    if r["mode"] == "fp16"
                ),
                None,
            )
            if base and fp16:
                lines.append(
                    f"Raw GEMM at n={max_n}: FP32 {base:.3f} TFLOPS, "
                    f"FP16 {fp16:.3f} TFLOPS "
                    f"({fp16/base:.2f}x)."
                )

    if valid_model:
        best = max(valid_model, key=lambda r: r["tokens_per_s"])
        if fp32_model and best["mode"] != "fp32":
            gain = best["tokens_per_s"] / fp32_model["tokens_per_s"]
            if gain > 1.05:
                lines.append(
                    f"Use {best['mode']} on this GPU for this "
                    f"workload; it is {gain:.2f}x faster than FP32."
                )
            else:
                lines.append(
                    "Keep FP32 on this GPU; no other mode is "
                    "meaningfully faster for this workload."
                )
        elif best["mode"] == "fp32":
            lines.append(
                "Keep FP32 on this GPU; it is the fastest measured "
                "training mode here."
            )
    elif not gemm:
        lines.append(
            "No benchmark completed; try a smaller --batch-size, "
            "--seq-len, or model size."
        )

    lines.append(
        "This verdict is specific to the GPU, driver, PyTorch/CUDA "
        "build, and shapes above. Re-run after driver or framework "
        "updates."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark FP32/FP16/BF16/TF32 raw GEMM and a GPT "
            "training step on the current CUDA GPU."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--batch-size", type=int, default=16,
                   help="model batch size (auto-reduced if OOM)")
    p.add_argument("--seq-len", type=int, default=512,
                   help="model context length")
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-kv-head", type=int, default=2)
    p.add_argument("--ffn-hidden", type=int, default=1408)
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--ce-chunk", type=int, default=64)
    p.add_argument("--no-checkpoint", action="store_true",
                   help="disable gradient checkpointing")
    p.add_argument("--steps", type=int, default=5,
                   help="timed model steps")
    p.add_argument("--model-warmup", type=int, default=2,
                   help="warmup model steps")
    p.add_argument("--gemm-iters", type=int, default=10)
    p.add_argument("--gemm-warmup", type=int, default=3)
    p.add_argument("--sizes", type=str, default="1024,2048,4096",
                   help="comma-separated square GEMM sizes")
    p.add_argument("--skip-gemm", action="store_true")
    p.add_argument("--skip-model", action="store_true")
    p.add_argument("--skip-native", action="store_true",
                   help="skip fp16_native/bf16_native model modes")
    p.add_argument("--compile", action="store_true",
                   help="benchmark torch.compile (compilation is not timed)")
    p.add_argument("--json", type=str, default=None,
                   help="write JSON results to this path, or '-' for stdout")
    p.add_argument("--quick", action="store_true",
                   help="smaller GEMMs and fewer steps")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.quick:
        args.sizes = "1024,2048"
        args.gemm_iters = 5
        args.gemm_warmup = 2
        args.steps = 3
        args.model_warmup = 1

    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        try:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        except Exception:
            pass

    info = collect_gpu_info()
    print_gpu_info(info)

    sizes: List[int] = []
    for item in args.sizes.split(","):
        item = item.strip()
        if not item:
            continue
        n = int(item)
        if n > 0:
            sizes.append(n)
    if not sizes and not args.skip_gemm:
        raise SystemExit("--sizes did not contain a positive integer")

    gemm_results: List[Dict[str, Any]] = []
    if not args.skip_gemm:
        print(
            "\nBenchmarking raw GEMM... "
            f"(iters={args.gemm_iters}, warmup={args.gemm_warmup})"
        )
        gemm_results = run_gemm(
            info,
            sizes,
            iters=args.gemm_iters,
            warmup=args.gemm_warmup,
        )
        print_gemm_table(gemm_results, sizes)

    model_results: List[Dict[str, Any]] = []
    if not args.skip_model:
        cfg = ModelConfig(
            vocab_size=args.vocab_size,
            block_size=args.seq_len,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_kv_head=args.n_kv_head,
            n_embd=args.d_model,
            ffn_hidden=args.ffn_hidden,
            grad_checkpointing=not args.no_checkpoint,
        )
        n_params = sum(
            p.numel() for p in GPT(cfg).parameters()
        )
        print(
            f"\nGPT benchmark: {n_params/1e6:.2f}M params, "
            f"T={cfg.block_size}, checkpoint={cfg.grad_checkpointing}"
        )
        batch = find_fitting_batch(
            cfg, args.batch_size, args.ce_chunk
        )
        if batch == 0:
            print(
                "Model benchmark skipped: even batch=1 OOMs. "
                "Try --seq-len 256 --d-model 256 --n-layer 4 "
                "or --no-checkpoint."
            )
        else:
            if batch != args.batch_size:
                print(
                    f"Batch auto-reduced from {args.batch_size} "
                    f"to {batch} to fit this GPU."
                )
            print(
                f"Using batch={batch}, seq_len={cfg.block_size} "
                f"({batch * cfg.block_size:,} tokens/step)"
            )
            model_results = run_model(
                info, cfg, batch, args
            )
            print_model_table(model_results)

    recommendation = build_recommendation(
        info, gemm_results, model_results
    )
    print("\n=== recommendation ===")
    print(recommendation)

    if args.json:
        payload = {
            "gpu": info,
            "settings": vars(args),
            "gemm": gemm_results,
            "model": model_results,
            "recommendation": recommendation,
        }
        text = json.dumps(payload, indent=2, default=str)
        if args.json == "-":
            print("\n=== json ===")
            print(text)
        else:
            with open(args.json, "w", encoding="utf-8") as f:
                f.write(text)
                f.write("\n")
            print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
