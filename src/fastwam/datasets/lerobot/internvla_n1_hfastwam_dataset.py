"""InternNav-style dataset that ALSO emits HFastWAM-VLM tensors.

Wraps :class:`internnav.dataset.internvla_n1_lerobot_dataset.NavPixelGoalDataset`
without modifying it (we add InternNav to sys.path so the import is a
runtime decision — InternNav stays one repo, FastWAM stays another).

Each ``__getitem__(i)`` returns a single dict that contains BOTH:

  ┌─ ChatML fields (from InternNav, unchanged) ──────────────┐
  │   input_ids        [S]                                    │
  │   labels           [S]   (instruction tokens = -100)      │
  │   position_ids     [3, 1, S]                              │
  │   attention_mask   [1] (effective length, packed format)  │
  │   pixel_values     [N_img_tokens, 3, ph, pw]              │
  │   image_grid_thw   [N_img, 3]                             │
  └───────────────────────────────────────────────────────────┘

  ┌─ HFastWAM fields (new, added on top) ────────────────────┐
  │   video            [3, fastwam_num_frames, H, W]          │
  │                       9 history + 8 future RGB, in [-1,1] │
  │   action           [predict_step_num, 4] (dx,dy,dθ,flag)  │
  │   action_is_pad    [predict_step_num] bool                │
  │   prompt           str  (raw instruction string)          │
  │   video_valid      bool — True iff 17 frames available    │
  │   action_valid     bool — True iff this sample has        │
  │                          continuous waypoints (pixel_goal │
  │                          subset). Stage1 sets λ_act=0     │
  │                          so this flag is informational.   │
  └───────────────────────────────────────────────────────────┘

Stage1 trainer ignores ``action`` and uses ``video`` for FM loss only
when ``video_valid``. Stage2 trainer uses ``action`` for FM loss only
when ``action_valid``.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
# torchcodec stub
# ---------------------------------------------------------------------------- #
# InternNav's lerobot dataset does ``from torchcodec.decoders import VideoDecoder``
# at module top-level. The fastwam conda env ships FFmpeg 8.0 (libavutil.so.60)
# which the installed torchcodec wheel does not support, so importing torchcodec
# raises a RuntimeError before we ever get a chance to use the decord fallback.
#
# Since our wrapper only reads single jpg frames (never decodes a real video
# file), we install a minimal stub *before* importing InternNav so the top-level
# import succeeds. If the user actually calls ``video_torchcodec(...)`` later,
# they'll get a clean AttributeError pointing here.
def _install_torchcodec_stub() -> None:
    if "torchcodec" in sys.modules:
        return
    import types

    class _StubVideoDecoder:  # pragma: no cover - never instantiated
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError(
                "torchcodec.decoders.VideoDecoder is stubbed out by FastWAM's "
                "internvla_n1_hfastwam_dataset wrapper because the fastwam "
                "conda env's FFmpeg 8.0 is incompatible with the installed "
                "torchcodec wheel. Our dataset only reads jpg frames and does "
                "not need it."
            )

    pkg = types.ModuleType("torchcodec")
    decoders_mod = types.ModuleType("torchcodec.decoders")
    decoders_mod.VideoDecoder = _StubVideoDecoder
    pkg.decoders = decoders_mod
    sys.modules["torchcodec"] = pkg
    sys.modules["torchcodec.decoders"] = decoders_mod


_install_torchcodec_stub()


# Add InternNav to sys.path so we can reuse its loader / preprocessor /
# trajectory utilities without copying them. InternNav itself stays
# untouched.
_INTERNNAV_ROOT = os.environ.get(
    "INTERNNAV_ROOT",
    "/apdcephfs_gy6/share_303214315/zhenye/code_vln/InternNav",
)
if _INTERNNAV_ROOT not in sys.path:
    sys.path.insert(0, _INTERNNAV_ROOT)


# ---------------------------------------------------------------------------- #
# Annotation cache loader (FASTWAM_ANNOTATION_CACHE)
# ---------------------------------------------------------------------------- #
# Why: ``get_annotations_from_lerobot_data`` reads N×scenes parquet files
# directly from cephfs every time the dataset is constructed. With 64 ranks
# loading 11 datasets, that's ~3 hours of cold cephfs reads dominated by
# stampede contention. ``scripts/cache_internnav_annotations.py`` pre-computes
# every (data_path, setting) combination on a single machine and pickles the
# result to disk; this function transparently swaps in those pickles when the
# env var ``FASTWAM_ANNOTATION_CACHE`` points to a directory that contains
# them, and otherwise falls through to the original (slow) loader.
def _install_annotation_cache() -> None:
    """Monkey-patch ``get_annotations_from_lerobot_data`` to load from pickle
    when ``FASTWAM_ANNOTATION_CACHE`` is set. Idempotent and safe to call
    multiple times.
    """
    cache_dir = os.environ.get("FASTWAM_ANNOTATION_CACHE")
    if not cache_dir:
        return
    if not os.path.isdir(cache_dir):
        logger.warning(
            "FASTWAM_ANNOTATION_CACHE=%s is not a directory; ignoring.",
            cache_dir,
        )
        return
    from internnav.dataset import internvla_n1_lerobot_dataset as _il

    if getattr(_il, "_fastwam_annotation_cache_installed", False):
        return

    _orig_get_anns = _il.get_annotations_from_lerobot_data
    data_dict = getattr(_il, "data_dict", {})

    # Build reverse lookup: (data_path, setting) -> dataset_key.
    # setting is `f"{height}cm_{pitch_2}deg"`.
    inverse = {}
    for k, info in data_dict.items():
        try:
            setting = f"{info['height']}cm_{info['pitch_2']}deg"
        except (KeyError, TypeError):
            continue
        inverse[(info["data_path"], setting)] = k

    import pickle as _pickle

    def _cached_get_anns(data_path, setting):
        key = inverse.get((data_path, setting))
        if key is not None:
            pkl_path = os.path.join(cache_dir, f"{key}.pkl")
            if os.path.exists(pkl_path):
                logger.warning(
                    "[FASTWAM_ANNOTATION_CACHE] hit: %s -> %s",
                    key, pkl_path,
                )
                with open(pkl_path, "rb") as f:
                    return _pickle.load(f)
            logger.warning(
                "[FASTWAM_ANNOTATION_CACHE] miss for key=%s (path=%s); "
                "falling back to slow loader.",
                key, pkl_path,
            )
        else:
            logger.warning(
                "[FASTWAM_ANNOTATION_CACHE] no key for (data_path=%s, "
                "setting=%s); falling back.",
                data_path, setting,
            )
        # Cache MISS → scan parquet directly. Prefer the tj5 fast-storage copy
        # (per-rank cephfs scan of gy6 under 64-rank load is the ~3 hr cold
        # start). Falls back to the original gy6 path when tj5 has no copy
        # (e.g. scalevln). The returned ``episode['video']`` paths embed
        # whichever root we scanned; the per-frame loader's tj5↔gy6 remap then
        # still applies at jpg-open time, so either root is safe downstream.
        scan_path = _remap_data_dir(data_path)
        if scan_path != data_path:
            logger.warning(
                "[FASTWAM_ANNOTATION_CACHE] scanning tj5 fast copy: %s", scan_path,
            )
        return _orig_get_anns(scan_path, setting)

    _il.get_annotations_from_lerobot_data = _cached_get_anns
    _il._fastwam_annotation_cache_installed = True
    logger.warning(
        "[FASTWAM_ANNOTATION_CACHE] installed cache loader for %d keys, dir=%s",
        len(inverse), cache_dir,
    )


# ---------------------------------------------------------------------------- #
# Debug scene limiter (FASTWAM_DEBUG_MAX_SCENES)
# ---------------------------------------------------------------------------- #
# Why: loading 60+ scenes from cephfs parquet takes minutes. During debug
# sessions only a handful of scenes is needed to exercise the data pipeline.
# Set FASTWAM_DEBUG_MAX_SCENES=5 to keep only the first N scenes.
#
# The cap happens BEFORE the ThreadPoolExecutor spins up parquet readers, so
# loading time is truly proportional to N (not to total scenes on disk).
# Technique: temporarily replace os.listdir in the InternNav module with a
# wrapper that truncates the scene-directory listing for the dataset root path.
# The original is restored via try/finally, so the patch is invisible outside
# the single call to get_annotations_from_lerobot_data.
def _install_debug_scene_limit() -> None:
    """Cap the number of scene dirs scanned by ``get_annotations_from_lerobot_data``.

    Set ``FASTWAM_DEBUG_MAX_SCENES=N`` (N > 0) to scan only the first N
    (sorted) scene directories, short-circuiting parquet loading for the rest.
    Idempotent.
    """
    max_scenes_str = os.environ.get("FASTWAM_DEBUG_MAX_SCENES", "0")
    try:
        max_scenes = int(max_scenes_str)
    except ValueError:
        logger.warning(
            "[FASTWAM_DEBUG_MAX_SCENES] invalid value %r (expected int); ignoring.",
            max_scenes_str,
        )
        return
    if max_scenes <= 0:
        return

    from internnav.dataset import internvla_n1_lerobot_dataset as _il

    if getattr(_il, "_fastwam_debug_scene_limit_installed", False):
        return

    _orig_fn = _il.get_annotations_from_lerobot_data

    def _limited(data_path, setting):
        # Save and temporarily replace os.listdir in the InternNav module so
        # the scene_ids list is truncated before any parquet I/O starts.
        _orig_listdir = _il.os.listdir

        def _capped_listdir(path):
            entries = _orig_listdir(path)
            if path == data_path:
                # Pre-filter to directories (mirrors the original list-comp)
                # then take the first max_scenes sorted entries.
                import os as _os
                all_dirs = sorted(
                    e for e in entries
                    if _os.path.isdir(_os.path.join(path, e))
                )
                limited = all_dirs[:max_scenes]
                logger.warning(
                    "[FASTWAM_DEBUG_MAX_SCENES=%d] scene_ids: %d → %d"
                    " (data_path=%s)",
                    max_scenes, len(all_dirs), len(limited), data_path,
                )
                # Returning only dirs is fine; the caller's own isdir-filter
                # is a no-op on a list that's already all-dirs.
                return limited
            return entries

        _il.os.listdir = _capped_listdir
        try:
            return _orig_fn(data_path, setting)
        finally:
            _il.os.listdir = _orig_listdir

    _il.get_annotations_from_lerobot_data = _limited
    _il._fastwam_debug_scene_limit_installed = True
    logger.warning(
        "[FASTWAM_DEBUG_MAX_SCENES=%d] scene limiter installed (listdir-cap fast path)",
        max_scenes,
    )


def _import_internnav():
    """Lazy-import InternNav modules so this file can be parsed without
    InternNav installed (handy for unit tests / docs builds).
    """
    from internnav.dataset.internvla_n1_lerobot_dataset import (
        NavPixelGoalDataset,
        get_trajectory_relative_to_frame,
        interpolate_and_resample_trajectory,
        clip_or_pad,
    )
    return {
        "NavPixelGoalDataset": NavPixelGoalDataset,
        "get_trajectory_relative_to_frame": get_trajectory_relative_to_frame,
        "interpolate_and_resample_trajectory": interpolate_and_resample_trajectory,
        "clip_or_pad": clip_or_pad,
    }


# ---------------------------------------------------------------------------- #
# Video-dir path remapping (fast-storage redirect)
# ---------------------------------------------------------------------------- #
# Default: redirect r2r and rxr from gy6 (slow, shared IO with training) to
# tj5 (separate storage cluster, ~45 MB/s on training nodes vs. timeout on gy6
# under 64-rank load). scalevln is redirected to each node's LOCAL /tmp/scalevln
# (rsync'd there for ~10x faster reads under high concurrency vs. gy6).
#
# Override via env var FASTWAM_DATA_PATH_REMAP="old1::new1;old2::new2"
# Set FASTWAM_DATA_PATH_REMAP="none" to disable all remapping.
_DEFAULT_PATH_REMAP: dict = {
    "/apdcephfs_gy6/share_303214315/jishengpeng/vlndata/InternData-N1/vln_ce/traj_data/r2r":
        "/tmp/r2r",
    "/apdcephfs_gy6/share_303214315/jishengpeng/vlndata/InternData-N1/vln_ce/traj_data/rxr":
        "/tmp/rxr",
    "/apdcephfs_gy6/share_303214315/jishengpeng/vlndata/InternData-N1/vln_ce/traj_data/scalevln":
        "/tmp/scalevln",
}


def _build_path_remap() -> dict:
    env_val = os.environ.get("FASTWAM_DATA_PATH_REMAP", "")
    if env_val.lower() == "none":
        return {}
    if env_val:
        remap: dict = {}
        for pair in env_val.split(";"):
            pair = pair.strip()
            if "::" in pair:
                old, new = pair.split("::", 1)
                remap[old.strip()] = new.strip()
        return remap
    # Default: use tj5 for r2r and rxr.
    return _DEFAULT_PATH_REMAP


_PATH_REMAP: dict = _build_path_remap()
if _PATH_REMAP:
    logger.warning(
        "[internvla_n1_hfastwam_dataset] video-dir path remap active (%d entries): %s",
        len(_PATH_REMAP),
        {k.split("/")[-1]: v.split("/")[-1] for k, v in _PATH_REMAP.items()},
    )


def _remap_video_path(path: str) -> str:
    """Replace slow-storage prefixes with fast-storage equivalents."""
    if not _PATH_REMAP:
        return path
    for old, new in _PATH_REMAP.items():
        if path.startswith(old):
            return new + path[len(old):]
    return path


# Per-sample frame-prefetch thread count. rxr samples need 22-24 frames; with
# only 4 threads a long sample's jpg decode serializes into a 6-10 s stall that,
# under 64-rank sync, balloons into a 200-470 s global slow step. Raising this
# parallelizes the per-sample reads. Tunable via env without code edits.
# tj5 (the fast copy r2r/rxr now read from) tolerates more concurrency than gy6.
_FRAME_PREFETCH_WORKERS = int(os.environ.get("FASTWAM_FRAME_PREFETCH_WORKERS", "12"))


def _remap_data_dir(path: str) -> str:
    """Directory-level tj5-first remap for *annotation* (parquet) scanning.

    Unlike :func:`_remap_video_path` (which always returns the tj5 prefix and
    relies on per-file gy6 fallback at open time), the annotation scanner does a
    single ``os.listdir(data_path)`` + per-episode ``parquet`` read against ONE
    root. So here we only switch to tj5 when the remapped directory actually
    exists on disk; otherwise we keep the original gy6 path. This gives
    "tj5 first, gy6 fallback" semantics at the dataset-root granularity.
    """
    remapped = _remap_video_path(path)
    if remapped != path and os.path.isdir(remapped):
        return remapped
    return path


def _load_rgb_jpg(path: str, size: int) -> torch.Tensor:
    """Load a single jpg as a tensor in [-1, 1], shape [3, size, size]."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # → [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3, H, W]


class InternVLAN1HFastWAMDataset(Dataset):
    """InternNav-style dual-stage dataset that also emits HFastWAM tensors.

    Constructed almost identically to :class:`NavPixelGoalDataset`. Pass
    the same ``tokenizer`` and ``data_args`` you would pass to InternNav.

    Additional ``data_args`` knobs (read with ``getattr`` / fallbacks):
        fastwam_video_size: int (default 224) — H=W for the raw RGB
            tensor fed to Wan2.2 VAE.
        fastwam_n_history_frames: int (default 9)
        fastwam_n_future_frames: int (default 8)
        fastwam_predict_step_num: int (default 8) — 4-dim action chunk
            length for HFastWAM's ActionDiT (kept independent of
            data_args.predict_step_num which still drives InternNav's
            traj_poses output).
    """

    def __init__(self, tokenizer, data_args):
        super().__init__()
        # ----- transformers 5.x compat shim ----- #
        # InternNav's preprocess_qwen_2_visual assumes
        # ``tokenizer.apply_chat_template(conv)`` returns a flat list of ints
        # (the transformers <=4.x default). transformers >=5.0 changed the
        # default to a BatchEncoding (dict-like), which breaks the
        # ``encode_id.copy(); target_mask[:3] = ...`` pattern at line ~265.
        # We patch the tokenizer to always return list[int] so InternNav's
        # code path stays untouched. Idempotent — only patches once.
        if not getattr(tokenizer, "_fastwam_apply_chat_template_patched", False):
            _orig_apply = tokenizer.apply_chat_template

            def _apply_chat_template_compat(*args, **kwargs):
                kwargs.setdefault("return_dict", False)
                out = _orig_apply(*args, **kwargs)
                if hasattr(out, "get") and "input_ids" in out:
                    out = out["input_ids"]
                if hasattr(out, "tolist"):
                    out = out.tolist()
                if isinstance(out, list) and out and isinstance(out[0], list):
                    out = out[0]
                return out

            tokenizer.apply_chat_template = _apply_chat_template_compat
            tokenizer._fastwam_apply_chat_template_patched = True

        # Install the annotation cache loader and optional debug scene limiter
        # BEFORE constructing the inner dataset, because NavPixelGoalDataset.__init__
        # calls get_annotations_from_lerobot_data inline.
        # Both patches are idempotent and no-ops when their env vars are unset.
        # Order matters: scene limiter wraps the (possibly cached) loader so the
        # chain is: NavPixelGoalDataset → _limited → _cached_get_anns → original.
        _install_annotation_cache()
        _install_debug_scene_limit()   # FASTWAM_DEBUG_MAX_SCENES=5 to limit scenes

        utils = _import_internnav()
        self._inner = utils["NavPixelGoalDataset"](tokenizer, data_args)
        self._get_trajectory_relative_to_frame = utils["get_trajectory_relative_to_frame"]
        self._interpolate_and_resample_trajectory = utils["interpolate_and_resample_trajectory"]
        self._clip_or_pad = utils["clip_or_pad"]

        # Filter out known-bad scenes (incomplete in source data, can't be fixed).
        # These scenes have <100 frames in ALL cameras even in gy6 — retry would
        # always fail/fallback and waste time.  Remove them from list_data_dict.
        _BAD_SCENE_SUBSTRINGS = os.environ.get(
            "FASTWAM_EXCLUDE_SCENES",
            "gZ6f7yhEvPG,YmJkqBEsHnH",  # r2r: 93 frames; rxr: no rgb
        ).split(",")
        if _BAD_SCENE_SUBSTRINGS and _BAD_SCENE_SUBSTRINGS[0]:
            before = len(self._inner.list_data_dict)
            self._inner.list_data_dict = [
                e for e in self._inner.list_data_dict
                if not any(bad in str(e[2] if len(e) > 2 else e) for bad in _BAD_SCENE_SUBSTRINGS)
            ]
            after = len(self._inner.list_data_dict)
            if before != after:
                logger.warning("[SCENE_FILTER] removed %d entries with bad scenes %s (before=%d after=%d)",
                               before - after, _BAD_SCENE_SUBSTRINGS, before, after)

        # Per-sample tokenization cache.
        # Two modes (checked in order):
        #   1. FASTWAM_QWEN_TOK_CACHE env → pre-computed .pt files on disk
        #      (built by scripts/cache_qwen_tokenization.py).  Zero compute.
        #   2. In-memory dict per worker — falls back to computing on first call,
        #      then caches for subsequent hits within the same worker lifetime.
        #      Max 2048 entries; LRU-style eviction keeps memory bounded.
        _tok_cache_dir = os.environ.get(
            "FASTWAM_QWEN_TOK_CACHE",
            "/tmp/qwen_tok_cache",   # node-local SSD — fastest reads
        )
        self._tok_cache_dir: "str | None" = _tok_cache_dir if os.path.isdir(_tok_cache_dir) else None
        _tok_cache_maxsize = int(os.environ.get("FASTWAM_TOK_CACHE_SIZE", "2048"))
        self._tok_cache: "dict" = {}         # in-memory hot cache (both modes)
        self._tok_cache_maxsize = _tok_cache_maxsize
        if self._tok_cache_dir:
            logger.warning("[TOK_CACHE_INIT] disk cache: %s (%d files)",
                           self._tok_cache_dir,
                           len(__import__('glob').glob(self._tok_cache_dir+'/*.pt')))
        else:
            logger.warning("[TOK_CACHE_INIT] disk cache NOT FOUND at %s — in-memory only", _tok_cache_dir)

        self.data_args = data_args
        self.tokenizer = tokenizer
        self.fastwam_video_size = int(getattr(data_args, "fastwam_video_size", 224))
        self.fastwam_n_history = int(getattr(data_args, "fastwam_n_history_frames", 9))
        self.fastwam_n_future = int(getattr(data_args, "fastwam_n_future_frames", 8))
        self.fastwam_total_frames = self.fastwam_n_history + self.fastwam_n_future
        self.fastwam_predict_step_num = int(
            getattr(data_args, "fastwam_predict_step_num", 8)
        )

        logger.info(
            "InternVLAN1HFastWAMDataset: %d samples, fastwam_video=%dx%d, "
            "frames=%d (%dh+%df), action_len=%d",
            len(self._inner),
            self.fastwam_video_size, self.fastwam_video_size,
            self.fastwam_total_frames,
            self.fastwam_n_history, self.fastwam_n_future,
            self.fastwam_predict_step_num,
        )

    # Expose modality_lengths/lengths so HF Trainer's length-based
    # samplers still work via the wrapper.
    def __len__(self) -> int:
        return len(self._inner)

    @property
    def modality_lengths(self):
        return self._inner.modality_lengths if hasattr(self._inner, "modality_lengths") else [1] * len(self)

    @property
    def lengths(self):
        return self._inner.lengths if hasattr(self._inner, "lengths") else [1] * len(self)

    def pre_calculated_length(self):
        if hasattr(self._inner, "pre_calculated_length"):
            return self._inner.pre_calculated_length()
        return np.array([1] * len(self))

    # ------------------------------------------------------------------ #
    # Item assembly
    # ------------------------------------------------------------------ #
    # Schema-required keys produced by InternNav's NavPixelGoalDataset.
    # If __getitem__ ever returns a dict missing any of these, it WILL
    # KeyError downstream in FlattenedDataCollatorForSupervisedDataset.__call__
    # (see internvla_n1_lerobot_dataset.py:1364). We retry such samples.
    _REQUIRED_CHATML_KEYS = (
        "input_ids", "labels", "position_ids", "attention_mask",
        "pixel_values", "image_grid_thw",
    )

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """Load one training sample with optimized I/O.

        InternNav's NavPixelGoalDataset.__getitem__ iterates
        ``range(0, end_frame_id)`` and opens **3 files per frame** (front-view
        jpg, lookdown jpg, depth png). For a pixel_goal sample with
        start_frame_id=10, end_frame_id=60 this is 180 file opens, but
        HFastWAM only ever needs ≤12:

          - history front-views    (≤ fastwam_n_history, all pitch_1)
          - current front-view     (1, pitch_1)
          - current lookdown       (1, pitch_2 — pixel_goal samples only,
                                    goes into VLM pixel_values as goal image)

        We install **two temporary patches** before each inner.__getitem__ call
        and restore them unconditionally in ``finally``:

        Patch A — ``end_frame_id`` clamp
            Replace ``list_data_dict[idx][7]`` with
            ``(start_frame_id, start_frame_id + 1)`` so InternNav's loop runs
            only frames ``0 … start_frame_id`` and never touches future frames.
            This eliminates the dominant I/O cost on long trajectories.

        Patch B — ``PIL.Image.open`` intercept
            Replace ``PIL.Image.open`` (i.e., the module-level ``Image.open``
            attribute that InternNav accesses at call time) with a thin wrapper
            that returns a 1×1 dummy image for:

            (a) Depth PNGs (``.png`` extension):  HFastWAM has no depth
                consumer; depth only fed ``traj_depths`` in InternNav's
                pixel_goal_only path, which we never enable.

            (b) Lookdown JPGs (``_{pitch_2}deg`` in path) for frames other
                than ``start_frame_id``:  history lookdowns are opened but
                never appended to ``images`` in InternNav's loop.  For
                turn/stop samples (``pose is None``) the current-frame
                lookdown is also unused.

        Both patches are **process-local**: DataLoader workers are independent
        forked processes; mutations to ``list_data_dict`` and ``Image.open``
        in one worker are invisible to others.  The ``finally`` clause ensures
        they cannot leak across retry attempts or between samples.

        I/O budget comparison (pixel_goal, start=10, end=60):
          Before optimisation : 60 iters × 3 files = 180 opens
          After optimisation  : 10 history + 1 current front + 1 current look
                                = 12 opens  (93 % reduction)
        """
        n = len(self._inner)
        last_exc: Optional[BaseException] = None
        _t0_getitem = time.perf_counter()

        for attempt in range(8):
            idx = i if attempt == 0 else (i + attempt * 31337) % n
            _t0_attempt = time.perf_counter()

            # ------------------------------------------------------------ #
            # Step 1: extract metadata BEFORE the inner call.
            # We need start_frame_id, pitch_2, and pose to build the patches.
            # ------------------------------------------------------------ #
            orig_entry = self._inner.list_data_dict[idx]
            try:
                (ep_id, data_path, video_dir, height, pitch_1, pitch_2,
                 instruction, (start_frame_id, end_frame_id),
                 action_label, pose) = orig_entry
            except Exception:
                # Newer InternNav layouts may extend the tuple — fall back to
                # positional access if available.
                try:
                    ep_id          = orig_entry[0]
                    data_path      = orig_entry[1]
                    video_dir      = orig_entry[2]
                    height         = orig_entry[3]
                    pitch_1        = orig_entry[4]
                    pitch_2        = orig_entry[5]
                    instruction    = orig_entry[6]
                    start_frame_id, end_frame_id = orig_entry[7]
                    action_label   = orig_entry[8]
                    pose           = orig_entry[9] if len(orig_entry) > 9 else None
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        "[InternVLAN1HFastWAMDataset] sample tuple unpack "
                        "failed at idx=%d: %s — retrying.", idx, exc,
                    )
                    continue

            # Remap video_dir from slow storage (gy6) to fast storage (tj5)
            # if FASTWAM_DATA_PATH_REMAP is configured.  data_path is kept as-is
            # because it's only used as an annotation-cache key (which was built
            # against the original gy6 paths); actual image loading uses video_dir.
            orig_video_dir = str(video_dir)   # keep gy6 path for _build_raw_video fallback
            video_dir = _remap_video_path(orig_video_dir)

            # ------------------------------------------------------------ #
            # Step 2: install I/O optimisation patches.
            # ------------------------------------------------------------ #

            # --- Patch A: clamp end_frame_id to skip all future frames ---
            fast_entry = list(orig_entry)
            fast_entry[7] = (int(start_frame_id), int(start_frame_id) + 1)
            self._inner.list_data_dict[idx] = tuple(fast_entry)

            # --- Patch B: intercept PIL.Image.open ----------------------
            # Capture everything needed by the closure as local variables so
            # the closure doesn't reference loop-iteration state incorrectly.
            _orig_open  = Image.open
            _pitch2_tag = f"_{pitch_2}deg"
            _start_fid  = int(start_frame_id)
            _has_pose   = (pose is not None)

            # Replicate InternNav's history_id computation so Patch B can
            # skip front-view opens for frames that InternNav will NOT use.
            # InternNav:  if start_frame_id != 0:
            #               history_id = np.unique(np.linspace(
            #                   0, start_frame_id-1, self.num_history, dtype=int32))
            #             else:
            #               history_id = []
            _num_history = getattr(self._inner, "num_history", self.fastwam_n_history)
            if _start_fid > 0:
                _needed_fids: "set[int]" = set(
                    int(x) for x in np.unique(
                        np.linspace(0, _start_fid - 1, _num_history, dtype=np.int32)
                    ).tolist()
                )
            else:
                _needed_fids = set()
            _needed_fids.add(_start_fid)  # current frame always needed

            # ----------------------------------------------------------------
            # IMPORTANT: expand _needed_fids to cover _build_raw_video's
            # frame set exactly so that _prefetch_one pre-loads everything and
            # _build_raw_video finds zero cache misses.
            #
            # Root-cause: InternNav uses linspace(0, start_fid-1, N) while
            # _build_raw_video uses linspace(0, start_fid, N) — different
            # endpoints → barely overlapping frame sets → 15 cache misses per
            # rxr sample → 15 extra gy6 reads in _build_raw_video → SLOW_LOAD.
            #
            # Fix: union in _build_raw_video's history frames AND its future
            # frames.  _fast_open is safe: it returns real images for all
            # frames in _needed_fids; InternNav only opens its own subset (all
            # included in this union); future frames are never opened by
            # InternNav (Patch A clamps end_frame_id = start_fid + 1).
            # ----------------------------------------------------------------
            if _start_fid >= 1:
                _needed_fids |= set(
                    int(x) for x in np.unique(
                        np.linspace(0, _start_fid, self.fastwam_n_history, dtype=np.int32)
                    ).tolist()
                )
            # Future frames (_build_raw_video requests start_fid+1 … end_fid)
            _end_fid = int(end_frame_id)
            _needed_fids |= set(
                range(_start_fid + 1, min(_end_fid, _start_fid + 1 + self.fastwam_n_future))
            )

            # --- Patch C: parallel pre-fetch of needed front-view frames ---
            # Opens all _needed_fids frames concurrently (ThreadPoolExecutor)
            # BEFORE InternNav's serial loop starts.  _fast_open then returns
            # from this cache, turning ~9 × 0.5 s serial reads into one
            # ~0.5 s parallel batch.  Also forwarded to _build_raw_video so
            # that history frames need not be re-opened there.
            _camera_dir = os.path.join(
                str(video_dir),
                f"observation.images.rgb.{int(height)}cm_{int(pitch_1)}deg",
            )

            def _prefetch_one(fid: int):
                # Try /tmp (local fast path) first.
                # If file missing (rsync not yet complete), fall back to original path.
                path = os.path.join(_camera_dir, f"episode_{int(ep_id):06d}_{fid}.jpg")
                _t_pf = time.perf_counter()
                try:
                    img = _orig_open(path)
                    img.load()
                    _pf_ms = (time.perf_counter() - _t_pf) * 1000
                    if _pf_ms > 200:
                        logger.warning("[PREFETCH_SLOW] fid=%d %.0fms path=%s", fid, _pf_ms, path)
                    return fid, img
                except FileNotFoundError:
                    # File missing on /tmp (rsync incomplete or source data gap).
                    # Do NOT fallback to gy6 (slow ~600ms). Return None so __getitem__
                    # retry picks a different sample. <1% of samples affected.
                    return fid, None
                except Exception:
                    return fid, None

            _prefetch_elapsed = 0.0
            _inner_elapsed = 0.0
            _build_video_elapsed = 0.0

            _prefetch_cache: Dict[int, Image.Image] = {}
            if _needed_fids:
                # Frame prefetch parallelism (FASTWAM_FRAME_PREFETCH_WORKERS,
                # default 12). r2r/rxr now read from tj5 (fast copy), which
                # tolerates higher concurrency than the old gy6 path; more
                # threads parallelize a long sample's 22-24 jpg decodes and
                # remove the 6-10 s stall that 64-rank sync amplifies.
                _pf_workers = min(len(_needed_fids), _FRAME_PREFETCH_WORKERS)
                _t0_prefetch = time.perf_counter()
                with ThreadPoolExecutor(max_workers=_pf_workers) as _pex:
                    for _fid, _img in _pex.map(_prefetch_one, sorted(_needed_fids)):
                        if _img is not None:
                            _prefetch_cache[_fid] = _img
                _prefetch_elapsed = time.perf_counter() - _t0_prefetch
                if _prefetch_elapsed > 5.0:
                    _ds_name = getattr(self._inner, "dataset_name", "?")
                    logger.warning(
                        "[SLOW_PREFETCH] prefetch=%.1fs ep=%s ds=%s "
                        "(%d frames, %d cache hits, %d workers)",
                        _prefetch_elapsed, ep_id, _ds_name,
                        len(_needed_fids), len(_prefetch_cache), _pf_workers,
                    )

            def _fast_open(path, *_a, **_kw):
                s = str(path)
                # Always remap gy6/tj5 paths to /tmp — no fallback to remote storage.
                # InternNav's list_data_dict stores the original gy6 paths; remap here
                # so ALL Image.open calls inside __getitem__ go to local /tmp.
                s_remapped = _remap_video_path(s)
                _t0_open = time.perf_counter()

                # (a) Depth images (.png) — HFastWAM has no depth consumer.
                if s.endswith(".png"):
                    return Image.new("L", (1, 1), 0)
                # (b) Lookdown camera (pitch_2) frames.
                if _pitch2_tag in s:
                    if not _has_pose:
                        return Image.new("RGB", (1, 1), 0)
                    try:
                        frame_num = int(s.rsplit("_", 1)[-1].split(".")[0])
                        if frame_num != _start_fid:
                            return Image.new("RGB", (1, 1), 0)
                    except (ValueError, IndexError):
                        pass
                    # Use remapped path for lookdown frame
                    try:
                        img = _orig_open(s_remapped, *_a, **_kw)
                        _e = time.perf_counter() - _t0_open
                        if _e > 0.5:
                            logger.warning("[SLOW_OPEN] lookdown %.2fs path=%s", _e, s_remapped)
                        return img
                    except FileNotFoundError:
                        # Missing lookdown frame — return dummy, __getitem__ retry handles it
                        return Image.new("RGB", (1, 1), 0)
                # (c) Front-view (pitch_1) frames
                try:
                    frame_num = int(s.rsplit("_", 1)[-1].split(".")[0])
                    if frame_num not in _needed_fids:
                        return Image.new("RGB", (1, 1), 0)
                    # Return pre-fetched image (zero latency).
                    if frame_num in _prefetch_cache:
                        return _prefetch_cache[frame_num]
                    # Cache miss — load from /tmp (remapped path).
                    # If /tmp file missing (rsync not yet complete), fall back to gy6.
                    try:
                        img = _orig_open(s_remapped, *_a, **_kw)
                        _e = time.perf_counter() - _t0_open
                        if _e > 0.5:
                            logger.warning("[SLOW_OPEN] cache_miss %.2fs frame=%d path=%s",
                                           _e, frame_num, s_remapped)
                        try:
                            img.load()
                            _prefetch_cache[frame_num] = img
                        except Exception:
                            pass
                        return img
                    except FileNotFoundError:
                        # /tmp incomplete — fall back to original path (gy6/tj5)
                        img = _orig_open(s, *_a, **_kw)
                        img.load()
                        return img
                except (ValueError, IndexError):
                    pass  # can't parse frame id — load from remapped path
                # Last resort: try /tmp first, fall back to original if missing
                try:
                    img = _orig_open(s_remapped, *_a, **_kw)
                    _e = time.perf_counter() - (time.perf_counter() - 0)
                    return img
                except FileNotFoundError:
                    return _orig_open(s, *_a, **_kw)

            Image.open = _fast_open

            # --- Patch D: eliminate per-image deepcopy of image_processor ---
            # InternNav's process_image_unified does copy.deepcopy(image_processor)
            # on every image call (9 images/sample × several hundred ms each =
            # the 6s SLOW_DATA we see). The deepcopy is defensive but unnecessary:
            # Qwen2VLImageProcessor.preprocess() does not mutate the processor.
            # Replace with a no-copy version for the duration of __getitem__.
            _orig_process_image = getattr(self._inner, "process_image_unified", None)
            if _orig_process_image is not None:
                _proc = getattr(
                    getattr(self._inner, "data_args", None), "image_processor", None
                )
                if _proc is not None:
                    import types as _types
                    _proc_call_times = []  # track per-image preprocess time
                    def _fast_process_image(img_or_path):
                        import PIL.Image as _PIL, time as _time
                        if isinstance(img_or_path, str):
                            image = _PIL.Image.open(img_or_path).convert("RGB")
                            MIN_SIZE = 28
                            w, h = image.size
                            if w < MIN_SIZE or h < MIN_SIZE:
                                image = image.resize(
                                    (max(w, MIN_SIZE), max(h, MIN_SIZE)), _PIL.Image.BILINEAR
                                )
                        else:
                            image = img_or_path
                        _tp = _time.perf_counter()
                        visual_processed = _proc.preprocess(image, return_tensors="pt")
                        _proc_call_times.append(_time.perf_counter() - _tp)
                        img_t = visual_processed["pixel_values"]
                        if isinstance(img_t, list):
                            img_t = img_t[0]
                        return img_t, visual_processed["image_grid_thw"][0]
                    self._inner.process_image_unified = _types.MethodType(
                        lambda self_inner, img: _fast_process_image(img),
                        self._inner,
                    )

            # ------------------------------------------------------------ #
            # Step 3: call InternNav's __getitem__ under the patches.
            #
            # Two-layer speedup for preprocess_qwen_2_visual (~3.8s/call):
            #
            # Patch E-1: skip copy.deepcopy(tokenizer) inside the function.
            #   The deepcopy costs 0.75s and only changes chat_template; we
            #   save+restore chat_template directly instead.
            #
            # Patch E-2: cache the tokenization result keyed by (ep_id,
            #   start_frame_id).  First call computes + stores; every subsequent
            # ------------------------------------------------------------ #
            _tok_cache = getattr(self, "_tok_cache", None)

            # Patch E-1: bypass copy.deepcopy(tokenizer) inside preprocess_qwen_2_visual.
            # The function deepcopies tokenizer only to set chat_template; we intercept
            # copy.deepcopy globally for the duration of __getitem__ and skip it for
            # tokenizer instances (returning the original object instead).
            import copy as _copy_mod
            from transformers import PreTrainedTokenizerBase as _TokBase
            _orig_copy_deepcopy = _copy_mod.deepcopy
            def _fast_deepcopy(obj, memo=None):
                if isinstance(obj, _TokBase):
                    return obj  # skip 0.75s deepcopy; caller only modifies chat_template
                return _orig_copy_deepcopy(obj, memo)
            _copy_mod.deepcopy = _fast_deepcopy


            _orig_preprocess_qwen = None
            _qwen_times = []
            _call_count = [0]

            try:
                _t0_inner = time.perf_counter()
                chatml = self._inner[idx]

                _inner_elapsed = time.perf_counter() - _t0_inner

                # ── Detailed timing breakdown (always log, not just on slow) ──
                _qwen_total   = sum(_qwen_times) if _qwen_times else 0.0
                _qwen_cached  = sum(1 for t in _qwen_times if t < 0.01) if _qwen_times else 0
                _proc_total   = sum(_proc_call_times) if _proc_call_times else 0.0
                _n_proc       = len(_proc_call_times)
                _unexplained  = max(0.0, _inner_elapsed - _proc_total
                                         - _prefetch_elapsed - _qwen_total)
                # disk hit: >0.001s (not mem) and <2.0s (not full compute ~3s)
                _qwen_disk    = sum(1 for t in _qwen_times if 0.001 < t < 2.0) if _qwen_times else 0
                _qwen_compute = len(_qwen_times) - _qwen_cached - _qwen_disk

                if _inner_elapsed > 1.0:   # log anything > 1s for detail
                    _ds_name = getattr(self._inner, "dataset_name", "?")
                    logger.warning(
                        "[TIMING] inner=%.3fs ep=%s idx=%d | "
                        "prefetch=%.3fs  img_proc=%.3fs(%d imgs)  "
                        "qwen_tok=%.3fs(%d calls: %d_mem %d_disk %d_compute)  "
                        "unexplained=%.3fs",
                        _inner_elapsed, ep_id, idx,
                        _prefetch_elapsed,
                        _proc_total, _n_proc,
                        _qwen_total, len(_qwen_times),
                        _qwen_cached, _qwen_disk, _qwen_compute,
                        _unexplained,
                    )
                elif _inner_elapsed > 0.1:  # fast path — brief log
                    logger.debug(
                        "[TIMING_FAST] inner=%.3fs qwen=%.3fs(%d_mem/%d_disk)",
                        _inner_elapsed, _qwen_total, _qwen_cached, _qwen_disk,
                    )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[InternVLAN1HFastWAMDataset] inner.__getitem__(%d) raised "
                    "%s: %s — retrying with idx=%d (attempt %d/8).",
                    idx, type(exc).__name__, exc,
                    (i + (attempt + 1) * 31337) % n, attempt,
                )
                continue
            finally:
                # Always restore — patches must never leak to other samples
                # or to _build_raw_video which needs the real Image.open.
                Image.open = _orig_open
                self._inner.list_data_dict[idx] = orig_entry
                # Restore copy.deepcopy (Patch E-1)
                _copy_mod.deepcopy = _orig_copy_deepcopy
                # Restore process_image_unified (Patch D)
                if _orig_process_image is not None:
                    self._inner.process_image_unified = _orig_process_image
                # Restore preprocess_qwen_2_visual (Patch E-2)
                if _orig_preprocess_qwen is not None:
                    import internnav.dataset.internvla_n1_lerobot_dataset as _inav_mod2
                    _inav_mod2.preprocess_qwen_2_visual = _orig_preprocess_qwen

            # ------------------------------------------------------------ #
            # Step 4: validate the ChatML dict returned by InternNav.
            # ------------------------------------------------------------ #
            if not isinstance(chatml, dict):
                logger.warning(
                    "[InternVLAN1HFastWAMDataset] inner.__getitem__(%d) returned "
                    "non-dict (%s) — retrying.",
                    idx, type(chatml).__name__,
                )
                continue
            missing = [k for k in self._REQUIRED_CHATML_KEYS if k not in chatml]
            if missing:
                logger.warning(
                    "[InternVLAN1HFastWAMDataset] inner.__getitem__(%d) returned "
                    "dict missing keys %s (got: %s) — retrying.",
                    idx, missing, sorted(chatml.keys()),
                )
                continue

            # ------------------------------------------------------------ #
            # Step 5: build HFastWAM-specific tensors.
            # Image.open is already restored at this point.  _build_raw_video
            # receives _prefetch_cache so history frames that were already
            # opened in parallel (Patch C) do not need a second cephfs round-
            # trip.  Future frames are opened in parallel inside _build_raw_video.
            # ------------------------------------------------------------ #
            try:
                _t0_build_video = time.perf_counter()
                video_tensor, video_valid = self._build_raw_video(
                    video_dir=video_dir,
                    fallback_video_dir=None,  # no gy6 fallback — /tmp is complete after repair
                    ep_id=int(ep_id),
                    height=int(height),
                    pitch_1=int(pitch_1),
                    start_frame_id=int(start_frame_id),
                    end_frame_id=int(end_frame_id),
                    frame_cache=_prefetch_cache,
                )
                _build_video_elapsed = time.perf_counter() - _t0_build_video

                # Action chunk for HFastWAM (only meaningful when pose is given).
                action_tensor, action_is_pad, action_valid = self._build_action_chunk(
                    pose=pose,
                    pitch_2=int(pitch_2),
                    action_label=action_label,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[InternVLAN1HFastWAMDataset] video/action build failed at "
                    "idx=%d: %s — retrying.", idx, exc,
                )
                continue

            chatml["video"]         = video_tensor          # [3, T, H, W]
            chatml["action"]        = action_tensor         # [predict_step_num, 4]
            chatml["action_is_pad"] = action_is_pad         # [predict_step_num]
            chatml["prompt"]        = str(instruction)
            chatml["video_valid"]   = torch.tensor(bool(video_valid))
            chatml["action_valid"]  = torch.tensor(bool(action_valid))
            # Sanity check: input_ids must survive the whole assembly.
            if "input_ids" not in chatml:
                logger.error(
                    "[InternVLAN1HFastWAMDataset] BUG: input_ids missing at return! "
                    "idx=%d keys=%s", idx, sorted(chatml.keys()),
                )
            _attempt_elapsed = time.perf_counter() - _t0_attempt
            if _attempt_elapsed > 30.0:
                _ds_name = getattr(self._inner, "dataset_name", "?")
                logger.warning(
                    "[SLOW_GETITEM] total=%.1fs ep=%s ds=%s idx=%d attempt=%d "
                    "(prefetch=%.1fs inner=%.1fs build_video=%.1fs)",
                    _attempt_elapsed, ep_id, _ds_name, idx, attempt,
                    _prefetch_elapsed, _inner_elapsed, _build_video_elapsed,
                )
            return chatml

        raise RuntimeError(
            f"InternVLAN1HFastWAMDataset: 8 attempts to load a valid sample "
            f"around idx={i} all failed; last exc: {last_exc}"
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _build_raw_video(
        self,
        *,
        video_dir: str,
        fallback_video_dir: "Optional[str]" = None,
        ep_id: int,
        height: int,
        pitch_1: int,
        start_frame_id: int,
        end_frame_id: int,
        frame_cache: "Optional[Dict[int, Image.Image]]" = None,
    ) -> Tuple[torch.Tensor, bool]:
        """Return ([3, T, H, W] tensor in [-1,1], video_valid: bool).

        Strategy:
          - pick ``fastwam_n_history`` evenly-spaced frames from
            ``[0, start_frame_id]`` (or zero-pad if not enough)
          - pick ``fastwam_n_future`` frames from
            ``[start_frame_id+1, end_frame_id]`` (or pad with last frame
            and mark video_valid=False)

        We use the ``pitch_1`` camera (forward-looking) to stay consistent
        with the FastWAM nav training setup.

        ``frame_cache`` (optional) maps frame_id → pre-opened PIL.Image.
        When provided (Patch C), history frames that were already fetched
        in parallel are reused here without a second cephfs round-trip.
        Remaining uncached frames (typically the 8 future frames) are opened
        in parallel via ThreadPoolExecutor.

        ``fallback_video_dir`` (optional) is the original un-remapped path
        (gy6).  When a tj5 frame open fails, we retry from this path so that
        Wan2.2 gets real frames instead of padded duplicates.
        """
        size = self.fastwam_video_size
        H_total = self.fastwam_n_history
        F_total = self.fastwam_n_future
        camera_dir = os.path.join(video_dir, f"observation.images.rgb.{height}cm_{pitch_1}deg")
        fallback_camera_dir: "Optional[str]" = (
            os.path.join(fallback_video_dir, f"observation.images.rgb.{height}cm_{pitch_1}deg")
            if fallback_video_dir and fallback_video_dir != video_dir
            else None
        )

        # ------------------------------------------------------------------ #
        # Helper: PIL.Image → float tensor in [-1, 1], shape [3, size, size]
        # ------------------------------------------------------------------ #
        def _pil_to_tensor(img: "Image.Image") -> torch.Tensor:
            img = img.convert("RGB").resize((size, size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        # ------------------------------------------------------------------ #
        # Compute all frame indices we need
        # ------------------------------------------------------------------ #
        if start_frame_id >= 1:
            history_ids = [
                int(x)
                for x in np.unique(
                    np.linspace(0, start_frame_id, H_total, dtype=np.int32)
                ).tolist()
            ]
        else:
            history_ids = [0]
        future_ids = list(
            range(start_frame_id + 1, min(end_frame_id, start_frame_id + 1 + F_total))
        )

        # Deduplicate while preserving order (history then future)
        all_ids_ordered = list(dict.fromkeys(history_ids + future_ids))

        # ------------------------------------------------------------------ #
        # Load all frames: cache hits → instant; misses → parallel IO
        # ------------------------------------------------------------------ #
        frame_dict: Dict[int, Optional[torch.Tensor]] = {}

        # Pass 1: resolve cache hits (zero IO cost)
        cache_miss_ids: List[int] = []
        for fid in all_ids_ordered:
            if frame_cache is not None and fid in frame_cache:
                try:
                    frame_dict[fid] = _pil_to_tensor(frame_cache[fid])
                except Exception:
                    frame_dict[fid] = None  # treat corrupt cache entry as miss
            else:
                cache_miss_ids.append(fid)

        # Pass 2: open cache-miss frames in parallel (no os.path.exists — just
        # try/except, saving one cephfs stat per file)
        def _load_one(fid: int) -> "Tuple[int, Optional[torch.Tensor]]":
            path = os.path.join(camera_dir, f"episode_{ep_id:06d}_{fid}.jpg")
            try:
                img = Image.open(path)
                img.load()
                return fid, _pil_to_tensor(img)
            except Exception:
                pass
            # tj5 miss (e.g. rxr future frames) → retry on gy6 fallback.
            # These 8 reads are issued in parallel via ThreadPoolExecutor so
            # they don't create a sequential bottleneck — unlike the history
            # frames in _fast_open which were sequential before Part-1 fixed
            # _prefetch_one.  Keeping this fallback preserves Wan2.2 FM
            # gradient signal for rxr (~40% of training data).
            if fallback_camera_dir is not None:
                fb_path = os.path.join(fallback_camera_dir, f"episode_{ep_id:06d}_{fid}.jpg")
                try:
                    img = Image.open(fb_path)
                    img.load()
                    return fid, _pil_to_tensor(img)
                except Exception:
                    pass
            return fid, None

        if cache_miss_ids:
            # Same parallelism knob as the prefetch path
            # (FASTWAM_FRAME_PREFETCH_WORKERS, default 12). These are the future
            # frames _build_raw_video still needs; parallelizing them removes the
            # tail stall on long rxr samples.
            _load_workers = min(len(cache_miss_ids), _FRAME_PREFETCH_WORKERS)
            _t0_load = time.perf_counter()
            with ThreadPoolExecutor(max_workers=_load_workers) as executor:
                for fid, tensor in executor.map(_load_one, cache_miss_ids):
                    frame_dict[fid] = tensor
            _load_elapsed = time.perf_counter() - _t0_load
            if _load_elapsed > 5.0:
                logger.warning(
                    "[SLOW_LOAD] _build_raw_video load=%.1fs ep=%d "
                    "(%d misses, %d workers)",
                    _load_elapsed, ep_id, len(cache_miss_ids), _load_workers,
                )

        # ------------------------------------------------------------------ #
        # Assemble history_padded
        # ------------------------------------------------------------------ #
        history_frames: List[Optional[torch.Tensor]] = [
            frame_dict.get(int(idx)) for idx in history_ids[-H_total:]
        ]
        # left-pad with first available frame or zeros
        first_real = next((f for f in history_frames if f is not None), None)
        if first_real is None:
            first_real = frame_dict.get(start_frame_id)
        history_padded = []
        for f in history_frames:
            if f is None and first_real is not None:
                history_padded.append(first_real)
            elif f is None:
                history_padded.append(torch.zeros(3, size, size))
            else:
                history_padded.append(f)
        while len(history_padded) < H_total:
            history_padded.insert(0, history_padded[0] if history_padded else torch.zeros(3, size, size))
        history_padded = history_padded[:H_total]

        # ------------------------------------------------------------------ #
        # Assemble future_padded
        # ------------------------------------------------------------------ #
        future_frames: List[Optional[torch.Tensor]] = [
            frame_dict.get(int(idx)) for idx in future_ids
        ]
        video_valid = (
            len([f for f in future_frames if f is not None]) >= F_total
            and len([f for f in history_padded if f is not None]) >= H_total
        )
        last_frame = (
            future_frames[-1]
            if future_frames and future_frames[-1] is not None
            else (history_padded[-1] if history_padded else torch.zeros(3, size, size))
        )
        future_padded = []
        for f in future_frames:
            future_padded.append(last_frame if f is None else f)
        while len(future_padded) < F_total:
            future_padded.append(last_frame)
        future_padded = future_padded[:F_total]

        all_frames = history_padded + future_padded  # T = H_total + F_total
        video = torch.stack(all_frames, dim=1)        # [3, T, H, W]
        return video, bool(video_valid)

    def _build_action_chunk(
        self,
        *,
        pose,
        pitch_2: int,
        action_label,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Map the InternNav sample into a [predict_step_num, 4] action.

        Returns ``(action, action_is_pad, action_valid)``:

          - pixel_goal sample (pose is not None):
                resample pose into ``predict_step_num`` waypoints
                (dx, dy, dθ), append ``moving_flag = 1``.
                ``action_is_pad`` is all False, ``action_valid = True``.
          - stop / turn sample (pose is None):
                action = zeros, ``moving_flag = 0`` everywhere.
                ``action_is_pad`` all True, ``action_valid = False``.
        """
        T = self.fastwam_predict_step_num
        if pose is None:
            zeros = torch.zeros(T, 4, dtype=torch.float32)
            pad = torch.ones(T, dtype=torch.bool)
            return zeros, pad, False

        try:
            rel_traj = self._get_trajectory_relative_to_frame(pose, camera_deg=pitch_2)
            _, rel_pose_resample = self._interpolate_and_resample_trajectory(rel_traj, T)
            rel_pose_resample = self._clip_or_pad(rel_pose_resample, T)  # [T, 3]
        except Exception as exc:
            logger.warning("Trajectory resample failed (%s); falling back to zeros.", exc)
            zeros = torch.zeros(T, 4, dtype=torch.float32)
            pad = torch.ones(T, dtype=torch.bool)
            return zeros, pad, False

        action_xyt = torch.from_numpy(np.asarray(rel_pose_resample, dtype=np.float32))  # [T, 3]
        # InternNav scales relative_pose[:, 0:2] *= 4 (see
        # interpolate_and_resample_trajectory). FastWAM nav_video_dataset
        # also normalises with the same factor, so we keep this scale.
        moving = torch.ones(T, 1, dtype=torch.float32)
        action = torch.cat([action_xyt, moving], dim=1)  # [T, 4]
        pad = torch.zeros(T, dtype=torch.bool)
        return action, pad, True


__all__ = ["InternVLAN1HFastWAMDataset"]
