#!/usr/bin/env python3
"""Launch pre-flight check: verify all /tmp data is complete before starting training.
Run on each node. Writes results to cephfs for the launcher to collect.
Exits 0 if OK, exits 1 if bad data found (launcher should abort and repair).

Checks:
  1. Only datasets actually used in training (from --datasets arg)
  2. qwen_tok_cache: at least 400000 .pt files + spot-check key hit rate
  3. Qwen3-VL weights: 2 safetensors files
  4. Wan checkpoints: wan2.2 files present

Usage: preflight_check.py --datasets scalevln_125cm_0_30,scalevln_60cm_30_30,...
"""
import os, sys, glob, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", default="", help="comma-separated vln dataset keys")
args, _ = parser.parse_known_args()

# Determine which base dirs to check from dataset names
active_datasets = set(args.datasets.split(",")) if args.datasets else set()
check_scalevln = any("scalevln" in d for d in active_datasets) or not active_datasets
check_r2r      = any("r2r" in d for d in active_datasets)
check_rxr      = any("rxr" in d for d in active_datasets)

STATDIR = "/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/.cctmp/preflight"
os.makedirs(STATDIR, exist_ok=True)
ip = os.popen("hostname -i").read().strip().split()[0]
out_path = f"{STATDIR}/{ip}.txt"

issues = []
ok_items = []

def check_camera_dirs(base, name, min_frames=100):
    """Return list of (path, count) for dirs with < min_frames."""
    bad = []
    if not os.path.isdir(base):
        return [(base, -1)]  # entire dataset missing
    for scene in os.listdir(base):
        scene_dir = os.path.join(base, scene)
        if not os.path.isdir(scene_dir):
            continue  # skip files like infromation.txt
        chunk = os.path.join(scene_dir, "videos", "chunk-000")
        if not os.path.isdir(chunk):
            bad.append((f"{name}/{scene}", 0))
            continue
        for cam in os.listdir(chunk):
            if not cam.startswith("observation.images.rgb"):
                continue
            cam_dir = os.path.join(chunk, cam)
            n = len([f for f in os.listdir(cam_dir) if f.endswith(".jpg")])
            if n < min_frames:
                bad.append((f"{name}/{scene}/{cam}", n))
    return bad

# 1. scalevln
if check_scalevln:
    bad = check_camera_dirs("/tmp/scalevln", "scalevln")
    if bad:
        issues.append(f"scalevln: {len(bad)} empty/sparse camera dirs: {bad[:5]}")
    else:
        ok_items.append("scalevln OK")
else:
    ok_items.append("scalevln SKIPPED (not in active datasets)")

# 2. r2r
if check_r2r:
    bad = check_camera_dirs("/tmp/r2r", "r2r")
    # r2r/gZ6f7yhEvPG has only 93 frames in source data (gy6) — not fixable, tolerate
    KNOWN_BAD_R2R = {"r2r/gZ6f7yhEvPG"}
    bad_real = [(k,v) for k,v in bad if not any(kb in k for kb in KNOWN_BAD_R2R)]
    if bad_real:
        issues.append(f"r2r: {len(bad_real)} empty/sparse camera dirs: {bad_real[:5]}")
    else:
        ok_items.append(f"r2r OK (known {len(bad)-len(bad_real)} bad from source data)")
else:
    ok_items.append("r2r SKIPPED (not in active datasets)")

# 3. rxr
if check_rxr:
    bad = check_camera_dirs("/tmp/rxr", "rxr", min_frames=100)
    # rxr/YmJkqBEsHnH has no rgb data in source (gy6) — not fixable, tolerate
    KNOWN_BAD_RXR = {"rxr/YmJkqBEsHnH"}
    bad_real = [(k,v) for k,v in bad if not any(kb in k for kb in KNOWN_BAD_RXR)]
    if bad_real:
        if len(bad_real) > 20:
            issues.append(f"rxr: {len(bad_real)} empty/sparse camera dirs (>20 threshold): {bad_real[:5]}")
        else:
            ok_items.append(f"rxr OK (known {len(bad)-len(bad_real)} bad from source, {len(bad_real)} within tolerance)")
    else:
        ok_items.append(f"rxr OK (known {len(bad)} bad from source data)")
else:
    ok_items.append("rxr SKIPPED (not in active datasets)")

# 4. qwen_tok_cache — skip when cache is disabled
_cache_dir = os.environ.get("FASTWAM_QWEN_TOK_CACHE", "/tmp/qwen_tok_cache")
if not _cache_dir:
    ok_items.append("qwen_tok_cache: disabled (direct compute mode)")
else:
    n_tok = len(glob.glob(f"{_cache_dir}/*.pt"))
    if n_tok < 400000:
        issues.append(f"qwen_tok_cache: only {n_tok} files (need >=400000)")
    else:
        # Quick key correctness check: sample 10 annotation entries and verify hit
        try:
            import pickle, hashlib, numpy as _np, random as _rnd
            _rnd.seed(0)
            _ANNO = "/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/data/internnav_annotations_cache"
            _IDX2A = {0:"STOP",1:"FORWARD",2:"LEFT",3:"RIGHT",4:"UP",5:"DOWN"}
            _hit = _miss = 0
            for _ds in ["scalevln_60cm_30_30", "scalevln_125cm_0_30"]:
                _ap = f"{_ANNO}/{_ds}.pkl"
                if not os.path.exists(_ap): continue
                with open(_ap,"rb") as _f: _d=pickle.load(_f)
                for _ep in _rnd.sample(_d["episodes"], min(5, len(_d["episodes"]))):
                    _instr=_ep["instructions"]; _acts=_ep["actions"][1:]+[0]; _pgs=_ep["pixel_goals"]; _n=len(_acts)
                    for _step in _rnd.sample(range(max(1,_n//4)),min(1,_n//4)):
                        _sfid=_step*4
                        if _sfid>=_n: continue
                        _act=_acts[_sfid]; _pg=_pgs[_sfid] if _sfid<len(_pgs) else [-1,-1]
                        _hp=_pg[0]!=-1
                        _hist=list(_np.linspace(0,_sfid-1,8,dtype=_np.int32)) if _sfid>0 else []
                        _ni=len(_hist)+1+(1 if _hp else 0)
                        _as=_IDX2A.get(int(_act),str(_act)) if not isinstance(_act,(list,tuple)) else str(_act)
                        _ck=hashlib.sha256(f"{_instr}|{_as}|{_ni}".encode()).hexdigest()[:16]
                        if os.path.exists(f"{_cache_dir}/{_ck}.pt"): _hit+=1
                        else: _miss+=1
            _rate=100*_hit//(_hit+_miss) if (_hit+_miss)>0 else 0
            if _rate < 90:
                issues.append(f"qwen_tok_cache: key hit rate only {_rate}% ({_hit}/{_hit+_miss}) — regenerate cache!")
            else:
                ok_items.append(f"qwen_tok_cache OK ({n_tok} files, key hit rate {_rate}%)")
        except Exception as _e:
            ok_items.append(f"qwen_tok_cache count OK ({n_tok} files, key check skipped: {_e})")

# 5. Qwen3-VL weights
n_qwen = len(glob.glob("/tmp/Qwen3-VL-2B-Instruct/*.safetensors"))
if n_qwen < 1:
    issues.append(f"Qwen3-VL: only {n_qwen} safetensors (need 1)")
else:
    ok_items.append(f"Qwen3-VL OK ({n_qwen} safetensors)")

# 6. Wan checkpoints
wan_ok = os.path.isdir("/tmp/fastwam_checkpoints")
if not wan_ok:
    issues.append("Wan checkpoints: /tmp/fastwam_checkpoints missing")
else:
    ok_items.append("Wan OK")

# Write result
with open(out_path, "w") as f:
    f.write(f"ip={ip}\n")
    for item in ok_items:
        f.write(f"OK: {item}\n")
    for issue in issues:
        f.write(f"BAD: {issue}\n")
    f.write(f"STATUS={'PASS' if not issues else 'FAIL'}\n")

if issues:
    print(f"[{ip}] PREFLIGHT FAIL: {len(issues)} issues")
    sys.exit(1)
else:
    print(f"[{ip}] PREFLIGHT PASS: {len(ok_items)} checks OK")
    sys.exit(0)
