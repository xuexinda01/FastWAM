"""Pre-cache InternNav lerobot annotations to disk.

Why: ``get_annotations_from_lerobot_data`` reads N×scenes parquet files
from cephfs. With 64 ranks doing this simultaneously, cephfs is the
bottleneck (~3 hours for our 11-dataset spec). Pre-computing once on a
single machine and pickling the result lets every rank start instantly.

Usage:
    python scripts/cache_internnav_annotations.py \
        --output /apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/data/internnav_annotations_cache

The script reads InternNav's ``DATASETS`` registry to find every
(data_path, setting) combination that our 11-dataset spec uses, then
pickles each combination to ``<output>/<dataset_key>.pkl``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add InternNav to sys.path
_INTERNNAV_ROOT = os.environ.get(
    "INTERNNAV_ROOT",
    "/apdcephfs_gy6/share_303214315/zhenye/code_vln/InternNav",
)
sys.path.insert(0, _INTERNNAV_ROOT)

# torchcodec stub before InternNav import
import types

if "torchcodec" not in sys.modules:
    pkg = types.ModuleType("torchcodec")
    decoders_mod = types.ModuleType("torchcodec.decoders")

    class _StubVideoDecoder:
        def __init__(self, *a, **k):
            raise RuntimeError("torchcodec stubbed for cache script")

    decoders_mod.VideoDecoder = _StubVideoDecoder
    pkg.decoders = decoders_mod
    sys.modules["torchcodec"] = pkg
    sys.modules["torchcodec.decoders"] = decoders_mod

# Import the function we want to call. We will also monkey-patch its
# inner ThreadPoolExecutor concurrency since on a single machine we can
# saturate cephfs better than the default 4 threads.
from internnav.dataset import internvla_n1_lerobot_dataset as _il
from internnav.dataset.internvla_n1_lerobot_dataset import (  # noqa: E402
    get_annotations_from_lerobot_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("cache_annotations")


# tj5-first remap for annotation scanning. Mirrors the dataset wrapper's
# _DEFAULT_PATH_REMAP. Build the cache by scanning the fast tj5 copy when it
# exists, falling back to the original gy6 path (e.g. scalevln, no tj5 copy).
_TJ5_REMAP = {
    "/apdcephfs_gy6/share_303214315/jishengpeng/vlndata/InternData-N1/vln_ce/traj_data/r2r":
        "/apdcephfs_tj5/share_302528826/xxd/r2r",
    "/apdcephfs_gy6/share_303214315/jishengpeng/vlndata/InternData-N1/vln_ce/traj_data/rxr":
        "/apdcephfs_tj5/share_302528826/xxd/rxr",
}


def _tj5_first_data_path(path: str) -> str:
    import os as _os
    for old, new in _TJ5_REMAP.items():
        if path.startswith(old):
            cand = new + path[len(old):]
            if _os.path.isdir(cand):
                return cand
    return path



def _resolve_dataset_specs() -> list[dict]:
    """Return the (data_path, setting, sampling_rate, key) for each dataset
    used by our stage1 launch script.

    We mirror the ``vln_dataset_use`` list. The actual per-key mapping
    lives inside InternNav's dataset registry; we re-derive it from the
    file directly so we don't drift.
    """
    # Dataset keys used by stage1.
    keys = [
        "r2r_125cm_0_30",
        "r2r_125cm_0_45",
        "r2r_60cm_15_15",
        "r2r_60cm_30_30",
        "rxr_125cm_0_30",
        "rxr_125cm_0_45",
        "rxr_60cm_15_15",
        "rxr_60cm_30_30",
        "scalevln_125cm_0_30",
        "scalevln_60cm_30_30",
        "scalevln_125cm_0_45",
    ]
    # Look up dataset registry inside InternNav. It's a module-level dict
    # mapping key → {data_path, height, pitch_1, pitch_2}.
    # The actual variable name is ``data_dict`` (verified in the source).
    DATASETS = getattr(_il, "data_dict", None)
    if DATASETS is None:
        for n in ("DATASETS", "VLN_DATASETS", "dataset_registry"):
            DATASETS = getattr(_il, n, None)
            if DATASETS is not None:
                break
    if DATASETS is None:
        raise RuntimeError(
            "Cannot find dataset registry in InternNav lerobot module. "
            "Check the variable name (we expected `data_dict`)."
        )

    out = []
    for k in keys:
        # k may match exactly or with a numerical sampling suffix like %30
        # (which our shell uses). Strip suffix if present.
        base = k.split("%", 1)[0]
        if base not in DATASETS:
            log.warning("Dataset key %s not in DATASETS registry; skipping.", base)
            continue
        info = DATASETS[base]
        height = info["height"]
        pitch_2 = info["pitch_2"]
        setting = f"{height}cm_{pitch_2}deg"
        out.append(
            {
                "key": base,
                "data_path": info["data_path"],
                "setting": setting,
                "info": info,
            }
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        default="/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/data/internnav_annotations_cache",
        help="Directory where per-dataset pickles will be written.",
    )
    p.add_argument(
        "--max_workers",
        type=int,
        default=24,
        help=(
            "Override the ThreadPoolExecutor max_workers used inside "
            "get_annotations_from_lerobot_data. Single-machine, we want "
            "more parallelism than the default 4."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-cache even if pickle already exists.",
    )
    p.add_argument(
        "--keys",
        type=str,
        default="",
        help=(
            "Comma-separated subset of dataset keys to process. Empty (default) "
            "means all 11. Useful for launching a second process in parallel "
            "with the main one to overlap I/O between disjoint datasets."
        ),
    )
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Monkey-patch ThreadPoolExecutor inside the lerobot module to use more
    # workers. The function does ``with ThreadPoolExecutor(max_workers=4)``,
    # so we wrap that constructor with our own that defaults to args.max_workers.
    real_TPE = ThreadPoolExecutor

    def _tpe_more_threads(max_workers=4, *a, **kw):
        max_workers = max(args.max_workers, int(max_workers or 4))
        return real_TPE(max_workers=max_workers, *a, **kw)

    _il.ThreadPoolExecutor = _tpe_more_threads
    log.info("Patched ThreadPoolExecutor: max_workers=%d", args.max_workers)

    specs = _resolve_dataset_specs()
    if args.keys.strip():
        wanted = {k.strip() for k in args.keys.split(",") if k.strip()}
        specs = [s for s in specs if s["key"] in wanted]
        log.info("--keys filter: keeping %d/%d datasets: %s",
                 len(specs), 11, sorted(s["key"] for s in specs))
    log.info("Resolved %d datasets to cache.", len(specs))

    grand_start = time.time()
    for idx, spec in enumerate(specs):
        out_path = out_dir / f"{spec['key']}.pkl"
        if out_path.exists() and not args.force:
            sz = out_path.stat().st_size / (1024 * 1024)
            log.info(
                "[%d/%d] %s already cached at %s (%.1f MB) — skip",
                idx + 1, len(specs), spec["key"], out_path, sz,
            )
            continue
        log.info(
            "[%d/%d] computing %s (data_path=%s setting=%s)",
            idx + 1, len(specs), spec["key"], spec["data_path"], spec["setting"],
        )
        t0 = time.time()
        scan_path = _tj5_first_data_path(spec["data_path"])
        if scan_path != spec["data_path"]:
            log.info("    using tj5 fast copy: %s", scan_path)
        anns = get_annotations_from_lerobot_data(scan_path, spec["setting"])
        n_eps = len(anns["episodes"])
        elapsed = time.time() - t0
        log.info(
            "[%d/%d] done %s: %d episodes in %.1f s",
            idx + 1, len(specs), spec["key"], n_eps, elapsed,
        )
        # Save pickle (atomic via rename)
        tmp = out_path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(anns, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.rename(out_path)
        sz = out_path.stat().st_size / (1024 * 1024)
        log.info("[%d/%d] saved %s (%.1f MB)", idx + 1, len(specs), out_path, sz)

    log.info(
        "Total wall: %.1f min. Cache dir: %s",
        (time.time() - grand_start) / 60.0,
        out_dir,
    )
    log.info("To use this cache, set env var FASTWAM_ANNOTATION_CACHE=%s", out_dir)


if __name__ == "__main__":
    main()
