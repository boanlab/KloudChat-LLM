"""Drop the Triton MLA decode kernel to a single pipeline stage on small-smem cards.

Why: vLLM's grouped MLA decode launcher already guards against shared-memory
overflow, but the guard keys off ``BLOCK_DMODEL`` alone::

    elif not is_hip_ and BLOCK_DMODEL >= 1024:
        num_stages = 1

For an MLA model with a 576-wide latent (``kv_lora_rank`` 512 + ``qk_rope_head_dim``
64 — GLM-4.7-Flash) the launcher splits that into ``BLOCK_DMODEL = 512`` and
``BLOCK_DPE = 64``. 512 < 1024, so the guard does not fire, ``num_stages`` stays 2,
and the kernel asks for 102400 B against GB10's 101376 B limit — over by exactly
1 KB. Engine init dies with::

    triton.runtime.errors.OutOfResources: out of resource: shared memory,
    Required: 102400, Hardware limit: 101376

It does not reproduce on datacenter parts (H100 offers ~228 KiB), which is why the
threshold was written that way.

What: force ``num_stages = 1`` for every MLA decode launch on NVIDIA. One stage
halves the smem request; the cost is less pipelining in a kernel the upstream
comment already describes as smem-light for MLA ("with is_mla there is only a
single c_kv in smem").

Idempotent — running twice is a no-op. Remove this patch once upstream sizes the
guard against the device limit rather than a fixed BLOCK_DMODEL threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/"
    "triton_decode_attention.py"
)

OLD = """    elif not is_hip_ and BLOCK_DMODEL >= 1024:"""
NEW = """    elif not is_hip_ and (BLOCK_DMODEL >= 1024 or is_mla):  # KloudChat: GB10 smem"""


def main() -> int:
    if not TARGET.is_file():
        print(f"[patch_triton_mla] {TARGET} not found — skipping", file=sys.stderr)
        return 0
    src = TARGET.read_text()
    if NEW in src:
        print("[patch_triton_mla] already applied")
        return 0
    if OLD not in src:
        # Upstream moved: fail loudly rather than silently shipping an image whose
        # MLA models crash-loop at engine init.
        print("[patch_triton_mla] anchor not found — upstream changed, review the patch",
              file=sys.stderr)
        return 1
    TARGET.write_text(src.replace(OLD, NEW, 1))
    print("[patch_triton_mla] applied: num_stages=1 for MLA decode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
