#!/usr/bin/env python3
"""Fast pre-training validation (<5s). No model loading, no heavy imports.

Checks:
1. Source code has all required patches (reads .py text, bypasses .pyc)
2. Disk cache has enough files + correct key hit rate
3. Sample data files exist in /tmp
"""
import sys, os, glob, hashlib, pickle, numpy as np, random

FASTWAM = "/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM"
CACHE   = os.environ.get("FASTWAM_QWEN_TOK_CACHE", "/tmp/qwen_tok_cache")
ANNO    = f"{FASTWAM}/data/internnav_annotations_cache"
SRC     = f"{FASTWAM}/src/fastwam/datasets/lerobot/internvla_n1_hfastwam_dataset.py"
IDX2A   = {0:"STOP",1:"FORWARD",2:"LEFT",3:"RIGHT",4:"UP",5:"DOWN"}

PASS = True
def check(name, ok, detail=""):
    global PASS
    sym = "OK" if ok else "FAIL"
    print(f"[{sym}] {name}{': ' + detail if detail else ''}")
    if not ok: PASS = False

# 1. Source code (reads .py directly, ignores stale .pyc)
with open(SRC) as f:
    src = f.read()
check("code: _tok_cache_dir init",      "_tok_cache_dir" in src)
check("code: Patch E-1 deepcopy bypass","_orig_copy_deepcopy" in src)
check("code: TIMING log",               "[TIMING]" in src)
check("code: /tmp remap in _fast_open", "s_remapped" in src)

# 2. Disk cache（cache 禁用时跳过）
n_files = len(glob.glob(f"{CACHE}/*.pt")) if CACHE else 0
cache_enabled = bool(CACHE)
check("cache: dir exists",     not cache_enabled or os.path.isdir(CACHE), CACHE or "(disabled)")
check("cache: >=400k files",   not cache_enabled or n_files >= 400000, f"{n_files}" if cache_enabled else "disabled")

# 3. Key hit rate（cache 禁用时跳过）
if cache_enabled and os.path.isdir(CACHE) and n_files >= 400000:
    random.seed(42)
    hit = miss = 0
    for ds in ["scalevln_60cm_30_30", "scalevln_125cm_0_30"]:
        ap = f"{ANNO}/{ds}.pkl"
        if not os.path.exists(ap): continue
        with open(ap, "rb") as f: data = pickle.load(f)
        for ep in random.sample(data["episodes"], min(15, len(data["episodes"]))):
            instr = ep["instructions"]
            acts  = ep["actions"][1:] + [0]
            pgs   = ep["pixel_goals"]
            n = len(acts)
            for step in random.sample(range(max(1, n//4)), min(2, max(1, n//4))):
                sfid = step * 4
                if sfid >= n: continue
                action = acts[sfid]
                pg = pgs[sfid] if sfid < len(pgs) else [-1,-1]
                has_pose = pg[0] != -1
                hist = list(np.linspace(0, sfid-1, 8, dtype=np.int32)) if sfid > 0 else []
                n_imgs = len(hist) + 1 + (1 if has_pose else 0)
                act_str = IDX2A.get(int(action), str(action)) if not isinstance(action,(list,tuple)) else str(action)
                key = hashlib.sha256(f"{instr}|{act_str}|{n_imgs}".encode()).hexdigest()[:16]
                if os.path.exists(f"{CACHE}/{key}.pt"): hit += 1
                else: miss += 1
    total = hit + miss
    rate  = 100 * hit // total if total else 0
    check("cache: key hit rate >=95%", rate >= 95, f"{hit}/{total}={rate}%")

# 4. Data files
random.seed(0)
for ds_base in ["/tmp/scalevln", "/tmp/r2r", "/tmp/rxr"]:
    if not os.path.isdir(ds_base): continue
    jpgs = glob.glob(f"{ds_base}/*/videos/chunk-000/observation.images.rgb.*/*.jpg")
    if not jpgs:
        check(f"data: {os.path.basename(ds_base)}", False, "no jpg files found")
        continue
    sample = random.sample(jpgs, min(10, len(jpgs)))
    missing = [f for f in sample if not os.path.exists(f)]
    check(f"data: {os.path.basename(ds_base)}",
          not missing, f"sample ok ({len(jpgs)} total)" if not missing else f"{len(missing)} missing")

print()
if PASS:
    print("ALL CHECKS PASSED")
    sys.exit(0)
else:
    print("VALIDATION FAILED")
    sys.exit(1)
