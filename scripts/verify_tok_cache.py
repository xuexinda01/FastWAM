#!/usr/bin/env python3
"""快速验证 cache 是否能命中，<10秒出结果，不需要启动训练。
测试: 随机采 100 个训练样本，模拟 _cached_preprocess_qwen 的 key 计算，
     看命中率。>95% 才算合格。
"""
import sys, types, os, hashlib, pickle, numpy as np, random, glob

# stub
pkg = types.ModuleType("torchcodec"); dm = types.ModuleType("torchcodec.decoders")
class S: pass
dm.VideoDecoder=S; pkg.decoders=dm
sys.modules["torchcodec"]=pkg; sys.modules["torchcodec.decoders"]=dm

CACHE = os.environ.get('FASTWAM_QWEN_TOK_CACHE', '/tmp/qwen_tok_cache')
ANNO  = '/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/data/internnav_annotations_cache'
IDX2A = {0:"STOP",1:"FORWARD",2:"LEFT",3:"RIGHT",4:"UP",5:"DOWN"}
NUM_HISTORY = 8
SAMPLE_STEP = 4

n_cache_files = len(glob.glob(f"{CACHE}/*.pt"))
print(f"Cache dir: {CACHE}")
print(f"Cache files: {n_cache_files}")
if n_cache_files < 1000:
    print("ERROR: cache is empty or too small! Run cache_qwen_tokenization.py first.")
    sys.exit(1)

random.seed(42)
total = hit = miss = 0
miss_examples = []

DATASETS = [
    ("scalevln_60cm_30_30", 60, 30, 30),
    ("scalevln_125cm_0_30", 125, 0, 30),
]

for ds_key, height, pitch_1, pitch_2 in DATASETS:
    anno_path = f"{ANNO}/{ds_key}.pkl"
    if not os.path.exists(anno_path): continue
    with open(anno_path, 'rb') as f:
        data = pickle.load(f)
    eps = data['episodes']
    sample_eps = random.sample(eps, min(25, len(eps)))

    for ep in sample_eps:
        instr = ep['instructions']
        acts  = ep['actions'][1:] + [0]
        pgs   = ep['pixel_goals']
        n = len(acts)

        # test 2 steps per episode
        for step in random.sample(range(n // SAMPLE_STEP), min(2, n // SAMPLE_STEP)):
            sfid = step * SAMPLE_STEP
            if sfid >= n: continue

            action = acts[sfid]
            pg = pgs[sfid] if sfid < len(pgs) else [-1,-1]
            has_pose = pg[0] != -1

            # --- Simulate InternNav's ACTUAL n_imgs (NO np.unique) ---
            if sfid > 0:
                hist = list(np.linspace(0, sfid-1, NUM_HISTORY, dtype=np.int32))
            else:
                hist = []
            n_imgs = len(hist) + 1 + (1 if has_pose else 0)

            act_str = IDX2A.get(int(action), str(action)) if not isinstance(action,(list,tuple)) else str(action)

            key = hashlib.sha256(f"{instr}|{act_str}|{n_imgs}".encode()).hexdigest()[:16]
            pt  = f"{CACHE}/{key}.pt"

            total += 1
            if os.path.exists(pt):
                hit += 1
            else:
                miss += 1
                if len(miss_examples) < 3:
                    miss_examples.append(f"  ds={ds_key} sfid={sfid} n_imgs={n_imgs} key={key}")

hit_rate = 100 * hit // total if total > 0 else 0
print(f"\nResult: {hit}/{total} hit = {hit_rate}% hit rate")
if miss_examples:
    print(f"Miss examples:")
    for e in miss_examples: print(e)

if hit_rate >= 95:
    print("\n✅ CACHE OK — safe to start training")
    sys.exit(0)
else:
    print(f"\n❌ CACHE BAD ({hit_rate}% < 95%) — DO NOT start training, regenerate cache first")
    sys.exit(1)
