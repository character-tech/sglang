# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Per-shape launch-config table for the Triton attention kernels.

This module centralizes every tuning-relevant launch knob (tile sizes,
``num_warps``, ``num_stages``, split policy, and the ROCm-only
``waves_per_eu`` / ``matrix_instr_nonkdim`` / ``kpack`` args) for the decode
and extend Triton kernels so the launch configs can be selected per Gemma-4
shape at runtime WITHOUT editing kernel code.

Resolution order (first hit wins), see :func:`lookup`:
  1. an entry in the env-pointed JSON file
     (``SGLANG_TRITON_ATTN_TUNING=/path/to.json``), if present and it has a
     matching key -- used by the sweep driver to inject a candidate config;
  2. a built-in override in :data:`_TABLE` for that shape bucket;
  3. the hardware default (:func:`_default_*`), which reproduces exactly the
     values the kernels launched with before this module existed.

The key is ``(kernel, head_dim, kv_group, kv_heads, bs_bucket, ctx_bucket)``
where ``kernel`` is one of ``"decode_stage1"``, ``"decode_stage2"``,
``"extend"``; ``bs_bucket`` / ``ctx_bucket`` are coarse buckets (see
:func:`bs_bucket` / :func:`ctx_bucket`) so a handful of measured points
generalize across the continuous batch-size / context ranges seen in serving.

Zero behavior change is guaranteed when neither an env-JSON nor a ``_TABLE``
entry matches: :func:`lookup` returns the default config, whose fields are the
literal constants the kernels used before (verified against
``decode_attention.py`` / ``extend_attention.py`` at the time of writing).
"""

import json
import logging
import os
from dataclasses import dataclass, replace
from typing import Optional

from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

_is_hip = is_hip()

# Env var pointing at a JSON file of overrides (sweep-time injection). The file
# is a list of {"key": [...], "config": {...}} objects; see _load_env_json.
_ENV_JSON = "SGLANG_TRITON_ATTN_TUNING"


@dataclass(frozen=True)
class LaunchConfig:
    """Resolved launch knobs for one kernel invocation.

    Fields default to ``None`` meaning "kernel picks its own value" — but the
    per-kernel default factories below always fill every field a given kernel
    consumes, so a config handed to a launch site is fully specified for that
    kernel. Fields a kernel does not use are left ``None`` and ignored.
    """

    # Tile / partitioning (decode stage1 + extend).
    block_n: Optional[int] = None
    block_h: Optional[int] = None  # decode stage1 only
    block_m: Optional[int] = None  # extend only
    # Split policy (decode). ``max_kv_splits`` is chosen by the backend from a
    # metadata budget; the table can cap it per shape via ``num_kv_splits_cap``
    # (None = use the backend value unchanged).
    num_kv_splits_cap: Optional[int] = None
    # Generic Triton launch args (all kernels).
    num_warps: Optional[int] = None
    num_stages: Optional[int] = None
    # ROCm-only launch args. Left None on CUDA (Triton rejects them there).
    waves_per_eu: Optional[int] = None
    matrix_instr_nonkdim: Optional[int] = None
    kpack: Optional[int] = None

    def extra_kargs(self) -> dict:
        """The ROCm ``extra_kargs`` dict for the Triton launch (empty on CUDA
        or when a field is None)."""
        out = {}
        if self.waves_per_eu is not None:
            out["waves_per_eu"] = self.waves_per_eu
        if self.matrix_instr_nonkdim is not None:
            out["matrix_instr_nonkdim"] = self.matrix_instr_nonkdim
        if self.kpack is not None:
            out["kpack"] = self.kpack
        return out


# --------------------------------------------------------------------------
# Bucketing. Coarse buckets keep the table small; the sweep measures at the
# bucket's representative point and the value applies across the bucket.
# --------------------------------------------------------------------------


def bs_bucket(bs: int) -> str:
    """Batch-size bucket. Boundaries chosen around the spec decode batch sizes
    {64, 256, 384, 768}."""
    if bs <= 96:
        return "s"  # small: ~64
    if bs <= 320:
        return "m"  # medium: ~256
    if bs <= 512:
        return "l"  # large: ~384
    return "xl"  # ~768


def ctx_bucket(ctx: int) -> str:
    """Mean-context bucket. Boundaries around the spec mean contexts
    {1k, 4.7k, 16k}; SWA is window-capped so it always lands in ``short``.

    Also used as the extend PREFIX bucket (mean prefix tokens per request:
    0 / 4.7k / 16k), so the runtime and sweep agree on slot 6 for extend.
    """
    if ctx <= 2048:
        return "short"  # ~1k (incl. SWA window 1024)
    if ctx <= 8192:
        return "mid"  # ~4.7k
    return "long"  # ~16k+


# Extend is keyed by per-LAUNCH shape, not batch size. The chunked-prefill
# budget (--chunked-prefill-size 6144/16384) is the TOTAL new-token count per
# prefill launch across all requests, and the launch's cost/best-tile depends
# on (total new tokens, how those tokens split across sequences, mean prefix).
# Slots 5/6 of the extend table key are therefore (ntok_split_label,
# prefix_bucket); both the sweep and the runtime derive them from the same
# signals (qo_indptr -> ntok + nseq; kv_indptr -> mean prefix).


def ntok_bucket(ntok: int) -> str:
    """Total-new-token bucket, around the two prod budgets {6144, 16384}."""
    return "6k" if ntok <= 11264 else "16k"  # split midway between 6k and 16k


def nseq_label(nseq: int) -> str:
    """How the launch's new tokens split across sequences. Runtime can only
    know the sequence count (qo_indptr length), so the label is a function of
    nseq alone and the sweep's split patterns (1 / 4 / 16 / ragged) map through
    this same function so keys line up."""
    if nseq <= 1:
        return "single"
    if nseq <= 8:
        return "few"
    return "many"


def extend_slot5(ntok: int, nseq: int) -> str:
    """Composite extend slot-5 label: ``<ntok_bucket>/<nseq_label>``."""
    return f"{ntok_bucket(ntok)}/{nseq_label(nseq)}"


# --------------------------------------------------------------------------
# Hardware defaults — MUST reproduce the pre-existing launch behavior exactly.
# --------------------------------------------------------------------------


def _default_decode_stage1(head_dim: int, kv_group: int) -> LaunchConfig:
    # From _decode_grouped_att_m_fwd: BLOCK_N=32 (16 if HIP and Lk>=576, which
    # neither 256 nor 512 hits), BLOCK_H=16, num_warps=4, num_stages=1 on HIP
    # (2 elsewhere), ROCm extra_kargs {waves_per_eu=1, nonkdim=16, kpack=2}.
    block_n = 16 if (_is_hip and head_dim >= 576) else 32
    if _is_hip:
        return LaunchConfig(
            block_n=block_n,
            block_h=16,
            num_warps=4,
            num_stages=1,
            waves_per_eu=1,
            matrix_instr_nonkdim=16,
            kpack=2,
        )
    return LaunchConfig(block_n=block_n, block_h=16, num_warps=4, num_stages=2)


def _default_decode_stage2() -> LaunchConfig:
    # From _decode_softmax_reducev_fwd: num_warps=4, num_stages=2, ROCm
    # extra_kargs {waves_per_eu=4, nonkdim=16, kpack=2}.
    if _is_hip:
        return LaunchConfig(
            num_warps=4,
            num_stages=2,
            waves_per_eu=4,
            matrix_instr_nonkdim=16,
            kpack=2,
        )
    return LaunchConfig(num_warps=4, num_stages=2)


def _default_extend(head_dim: int, kv_group: int) -> LaunchConfig:
    """Extend defaults. On HIP (the tuning target) reproduce
    ``_get_block_sizes_for_extend_attention`` + the prior extend launch
    constants. On CUDA, defer BLOCK_M/N/num_warps to that helper (arch-branchy)
    by leaving them ``None`` so the launch site keeps the helper's values —
    guaranteeing zero CUDA behavior change.
    """
    # Import here (not at module top) to avoid a hard dep for CUDA-only users.
    from sglang.kernels.ops.attention.extend_attention import (
        _get_block_sizes_for_extend_attention,
    )

    _, _, _, block_m, block_n, num_warps = _get_block_sizes_for_extend_attention(
        head_dim, head_dim
    )
    if _is_hip:
        return LaunchConfig(
            block_m=block_m,
            block_n=block_n,
            num_warps=num_warps,
            num_stages=1,
            waves_per_eu=1,
            matrix_instr_nonkdim=16,
            kpack=2,
        )
    return LaunchConfig(
        block_m=block_m, block_n=block_n, num_warps=num_warps, num_stages=1
    )


def _default_for(kernel: str, head_dim: int, kv_group: int) -> LaunchConfig:
    if kernel == "decode_stage1":
        return _default_decode_stage1(head_dim, kv_group)
    if kernel == "decode_stage2":
        return _default_decode_stage2()
    if kernel == "extend":
        return _default_extend(head_dim, kv_group)
    raise ValueError(f"unknown kernel {kernel!r}")


# --------------------------------------------------------------------------
# Built-in overrides. EMPTY by default — populate from sweep output (the
# driver's --write-table mode emits entries in this shape). Any entry here
# changes runtime behavior for the matching bucket, so it must be a measured
# win, not a guess.
#
# Key layout matches _key(): (kernel, head_dim, kv_group, kv_heads,
# bs_bucket, ctx_bucket). Use None in a slot to wildcard it (matched only if
# no fully-specified key hits — see lookup()).
# --------------------------------------------------------------------------

_TABLE: "dict[tuple, dict]" = {}

# Built-in per-shape best configs, measured on MI325X (gfx942) for the
# gemma-4 attention shapes (sweep record: HANDOFF 11.17 / CAI-G4-ATTN.md 9.5).
# Shipped as a .py module (the image's source-overlay rsync only carries
# *.py/*.hip/*.cpp/*.h - a .json here silently vanishes from the image).
# Same entry format as the env JSON; the env JSON still wins over these.
if _is_hip:
    try:
        from sglang.kernels.ops.attention.gemma4_mi325x_tuning import (
            TABLE as _BUNDLED_TABLE,
        )
        for _item in _BUNDLED_TABLE:
            _TABLE[tuple(_item["key"])] = _item["config"]
        logger.info("triton-attn tuning table: %d bundled entries loaded",
                    len(_TABLE))
    except ImportError as _e:
        logger.warning("bundled tuning table not importable: %s", _e)


def _key(kernel, head_dim, kv_group, kv_heads, bs_b, ctx_b):
    return (kernel, head_dim, kv_group, kv_heads, bs_b, ctx_b)


# --------------------------------------------------------------------------
# Env-JSON overrides (sweep-time injection). Parsed once and cached.
# --------------------------------------------------------------------------

_env_json_cache = None
_env_json_path = None


def _load_env_json():
    """Load and cache the SGLANG_TRITON_ATTN_TUNING JSON, re-reading if the
    env var changes (the sweep driver rewrites it between configs).

    Returns a dict mapping the 6-tuple key to a field dict, or {} if unset.
    """
    global _env_json_cache, _env_json_path
    path = os.environ.get(_ENV_JSON)
    if path == _env_json_path and _env_json_cache is not None:
        return _env_json_cache
    _env_json_path = path
    if not path:
        _env_json_cache = {}
        return _env_json_cache
    try:
        with open(path) as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning("could not read %s=%s: %s", _ENV_JSON, path, e)
        _env_json_cache = {}
        return _env_json_cache
    table = {}
    for item in raw:
        k = tuple(item["key"])
        table[k] = item["config"]
    _env_json_cache = table
    logger.info("loaded %d triton-attn tuning override(s) from %s", len(table), path)
    return _env_json_cache


def _apply_override(cfg: LaunchConfig, fields: dict) -> LaunchConfig:
    """Return ``cfg`` with the (non-None) fields from ``fields`` applied.

    Unknown keys are ignored so a JSON emitted by a newer sweep driver does
    not crash an older runtime.
    """
    known = {f.name for f in LaunchConfig.__dataclass_fields__.values()}
    upd = {k: v for k, v in fields.items() if k in known and v is not None}
    return replace(cfg, **upd) if upd else cfg


def lookup(
    kernel: str,
    head_dim: int,
    kv_group: int,
    kv_heads: int,
    bs: int,
    ctx: int,
    ntok: int = 0,
    nseq: int = 0,
) -> LaunchConfig:
    """Resolve the launch config for one kernel invocation.

    For decode, ``bs``/``ctx`` are the batch size and mean context, bucketed
    into key slots 5/6. For extend, the launch is keyed by per-LAUNCH shape:
    pass ``ntok`` (total new tokens), ``nseq`` (sequence count), and ``ctx``
    (mean prefix) — slot 5 becomes ``<ntok_bucket>/<nseq_label>`` and slot 6
    the prefix bucket. Returns a fully-specified :class:`LaunchConfig`; with
    no override present it is byte-for-byte the pre-existing default.
    """
    cfg = _default_for(kernel, head_dim, kv_group)
    if kernel == "extend":
        slot5 = extend_slot5(ntok, nseq)
        slot6 = ctx_bucket(ctx)  # mean-prefix bucket
    else:
        slot5 = bs_bucket(bs)
        slot6 = ctx_bucket(ctx)
    bs_b, ctx_b = slot5, slot6

    # 2 (built-in table) then 1 (env-JSON) — env wins, so apply table first.
    for k in (
        _key(kernel, head_dim, kv_group, kv_heads, bs_b, ctx_b),
        _key(kernel, head_dim, kv_group, kv_heads, bs_b, None),
        _key(kernel, head_dim, kv_group, kv_heads, None, None),
    ):
        if k in _TABLE:
            cfg = _apply_override(cfg, _TABLE[k])
            break

    env = _load_env_json()
    if env:
        for k in (
            _key(kernel, head_dim, kv_group, kv_heads, bs_b, ctx_b),
            _key(kernel, head_dim, kv_group, kv_heads, bs_b, None),
            _key(kernel, head_dim, kv_group, kv_heads, None, None),
        ):
            if k in env:
                cfg = _apply_override(cfg, env[k])
                break

    return cfg
