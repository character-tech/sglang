# SPDX-License-Identifier: Apache-2.0
"""Torch caching-allocator guard: hard reserved-memory cap + stats reporter.

Why the cap exists: the torch caching allocator never returns memory to the
device on its own — under sustained varied-shape load its reserved high-water
parks at ~all free VRAM (it self-recovers from its OWN allocation failures by
flushing cached blocks and retrying, so it operates at the wall indefinitely).
But ROCm's lazy per-queue scratch commits (first dispatch of any kernel with
register spills; sized for every scratch slot on the device — GB-scale on
gfx942) happen OUTSIDE hipMalloc on an HSA runtime thread and CANNOT reclaim
torch's cache. If they land while torch is parked at the wall, the process
dies with an uncatchable HSA_STATUS_ERROR_OUT_OF_RESOURCES queue-callback
abort (ROCm/clr#281; the 2026-08-25 gemma-4 mixed-chunk incident, 7/10 pods).

SGLANG_TORCH_MEM_FRACTION_CAP keeps (1 - cap) of the device permanently out
of torch's reach via torch.cuda.set_per_process_memory_fraction (caps total
RESERVED segments; ROCm-supported), so runtime-internal allocations always
find free VRAM. Torch pressure at the cap resolves through its internal
cache-flush-and-retry, then a CATCHABLE torch.OutOfMemoryError at worst.
Setting the cap also arms PYTORCH_ALLOC_CONF=garbage_collection_threshold:X,
which is documented-but-inert without a fraction cap.

SGLANG_TORCH_MEM_STATS_INTERVAL_S > 0 starts a daemon thread logging
allocator peaks (allocated = true tensor demand; reserved = what torch takes
from the device; num_alloc_retries = times the wall was touched) plus
driver-level free memory — the instruments for sizing mem-fraction-static
from data instead of rocm-smi's saturating "used" metric.
"""

import logging
import threading
import time

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def maybe_apply_torch_mem_cap(*, gpu_id: int) -> None:
    """Apply the env-configured allocator cap and start the stats reporter."""
    cap = envs.SGLANG_TORCH_MEM_FRACTION_CAP.get()
    if cap is not None:
        torch.cuda.set_per_process_memory_fraction(float(cap), gpu_id)
        total = torch.cuda.get_device_properties(gpu_id).total_memory
        logger.info(
            "torch allocator cap ACTIVE: %.3f of %.1f GiB (reserved for "
            "non-torch runtime: %.1f GiB)",
            cap,
            total / (1 << 30),
            (1 - cap) * total / (1 << 30),
        )
    interval_s = envs.SGLANG_TORCH_MEM_STATS_INTERVAL_S.get()
    if interval_s and interval_s > 0:
        thread = threading.Thread(
            target=_stats_reporter_loop,
            kwargs={"gpu_id": gpu_id, "interval_s": float(interval_s)},
            daemon=True,
            name="torch-mem-stats",
        )
        thread.start()


def _stats_reporter_loop(*, gpu_id: int, interval_s: float) -> None:
    while True:
        time.sleep(interval_s)
        try:
            stats = torch.cuda.memory_stats(gpu_id)
            free, total = torch.cuda.mem_get_info(gpu_id)
            logger.info(
                "torch-mem-stats: alloc_cur=%.1f alloc_peak=%.1f "
                "reserved_cur=%.1f reserved_peak=%.1f GiB "
                "alloc_retries=%d ooms=%d driver_free=%.1f GiB",
                stats.get("allocated_bytes.all.current", 0) / (1 << 30),
                stats.get("allocated_bytes.all.peak", 0) / (1 << 30),
                stats.get("reserved_bytes.all.current", 0) / (1 << 30),
                stats.get("reserved_bytes.all.peak", 0) / (1 << 30),
                stats.get("num_alloc_retries", 0),
                stats.get("num_ooms", 0),
                free / (1 << 30),
            )
        except Exception as e:
            logger.warning("torch-mem-stats reporter failed: %s", e)
            return
