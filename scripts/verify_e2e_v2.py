"""
End-to-end sanity check for the NEW yaw-aware training+inference pipeline.

Pipeline simulated here:
  dataset (normalized action label)
    → "model output" (we just take the label as if model predicted perfectly)
    → fastwam_agent.predict_trajectory denorm + left→right flip
    → traj_utils.fastwam_traj_to_actions
    → discrete Habitat action sequence

For each sample category we expect:
  FWD          → mostly FORWARD, few/no LEFT/RIGHT
  SMALL_TURN   → some LEFT/RIGHT + some FORWARD
  BIG_TURN     → many LEFT or RIGHT (the dominant direction)
  TERMINAL     → if stop-flag majority: STOP; else: short sequence

Note: We bypass the actual model. This proves the data pipeline is internally
consistent. Real model output adds noise but layout/sign should match.
"""
import os
import sys
import numpy as np

PROJECT_ROOT = "/apdcephfs/wx_feature/home/xxd/FastWAM"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, "/apdcephfs/wx_feature/home/xxd/fast-eval")

from visualize_nav_samples import build_dataset  # noqa: E402
from traj_utils import fastwam_traj_to_actions  # noqa: E402

# Same constants as fastwam_agent.predict_trajectory
ACTION_SCALE = np.array([0.2504, 0.2165, 0.2625], dtype=np.float32)


def model_to_eval(action_normalized_with_flag: np.ndarray) -> np.ndarray:
    """Replicate fastwam_agent.predict_trajectory's adapter: denorm + flip."""
    out = action_normalized_with_flag.astype(np.float64).copy()
    # Denormalize first 3 dims
    out[:, 0] *= ACTION_SCALE[0]    # forward
    out[:, 1] *= ACTION_SCALE[1]    # left (still left at this point)
    out[:, 2] *= ACTION_SCALE[2]    # yaw
    # Flip left → right
    out[:, 1] *= -1.0
    return out


def summarize(a_list):
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for a in a_list:
        counts[a] += 1
    return f"FWD={counts[1]:>2} L={counts[2]:>2} R={counts[3]:>2} STOP={counts[0]:>2}"


def main():
    ds = build_dataset("/apdcephfs/wx_feature/home/xxd/debugdata/debug_data")

    # collect a few samples per category
    cat_to_idx = {}
    for s_idx, c in enumerate(ds.sample_categories):
        cat_to_idx.setdefault(c, []).append(s_idx)
    print(f"Categories available: {[(c, len(v)) for c, v in cat_to_idx.items()]}")
    print()

    print(f"{'CAT':>11} {'idx':>5}  "
          f"{'a0_total':>9} {'a1_total':>9} {'a2_total':>9}  "
          f"{'discrete actions (denorm + flip → traj_utils)':<60}")
    print("-" * 130)

    total_per_cat = {}
    issued_per_cat = {}

    for cat in ["FWD", "SMALL_TURN", "BIG_TURN", "TERMINAL", "OTHER"]:
        if cat not in cat_to_idx:
            continue
        # take 3 samples from this category
        for idx in cat_to_idx[cat][:3]:
            data = ds._get(idx)
            action_norm = data["action"].numpy()  # (T, 4)
            # Translate to "model output" → eval space
            traj = model_to_eval(action_norm)
            a_list = fastwam_traj_to_actions(traj, step_size=0.25,
                                             turn_angle_deg=15)
            a0t, a1t, a2t = traj[:, 0].sum(), traj[:, 1].sum(), traj[:, 2].sum()
            print(f"{cat:>11} {idx:>5}  {a0t:+9.3f} {a1t:+9.3f} {a2t:+9.3f}  "
                  f"{summarize(a_list)}")
            total_per_cat[cat] = total_per_cat.get(cat, 0) + 1
            issued_per_cat.setdefault(cat, {0: 0, 1: 0, 2: 0, 3: 0})
            for a in a_list:
                issued_per_cat[cat][a] += 1
        print()

    print()
    print("=" * 78)
    print(f"{'CAT':>11}  {'samples':>7}  total actions issued")
    print("-" * 78)
    for cat, counts in issued_per_cat.items():
        total = sum(counts.values())
        print(f"{cat:>11}  {total_per_cat[cat]:>7}  "
              f"FWD={counts[1]:>3} L={counts[2]:>3} R={counts[3]:>3} STOP={counts[0]:>3}  "
              f"(per-sample avg: {total/total_per_cat[cat]:.1f})")
    print("=" * 78)


if __name__ == "__main__":
    main()
