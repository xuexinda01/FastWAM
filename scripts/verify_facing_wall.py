"""
Simulate what happens when the model is asked to handle a "facing-the-wall"
situation. We bypass the actual model and feed traj_utils a *hypothetical*
trajectory that represents what each scenario logically requires.

Scenarios:
  S1 — Directly facing wall, MUST turn 180°:
        ground truth: 0 forward motion for several steps (turning), then
        forward motion in the OPPOSITE direction. But xy_to_delta_xyt's
        spline pipeline would have collapsed these into a near-zero
        trajectory; that's what the model is most likely to produce.
        We test: model outputs essentially zero motion.

  S2 — Wall slightly in front (need slight left turn):
        model outputs a small arc to the left.

  S3 — IDEAL behavior (what we WISH the model would do for S1):
        model outputs trajectory that goes BACKWARD (negative dim0).
        We test: does traj_utils correctly issue 12 LEFT actions?

  S4 — IDEAL behavior, 90° left turn needed:
        model outputs trajectory that goes leftward (negative dim1
        before flip = +right after flip, but ideally model would output
        -dim1 = +left = ... wait no, training is +left). So:
        model output: dim0=0, dim1=+0.5 (clear leftward shift)

For each scenario we run the full chain:
    (raw model output)
        -> /=4
        -> dim1 *= -1     (current adapter)
        -> fastwam_traj_to_actions
"""
import os
import sys
import numpy as np

PROJECT_ROOT = "/apdcephfs/wx_feature/home/xxd/FastWAM"
sys.path.insert(0, "/apdcephfs/wx_feature/home/xxd/fast-eval")

from traj_utils import fastwam_traj_to_actions, STOP, FORWARD, LEFT, RIGHT  # noqa: E402

NAMES = {0: "STOP", 1: "FWD", 2: "LEFT", 3: "RIGHT"}


def run(label, raw_model_out):
    """raw_model_out: [T, 3] in TRAINING convention (forward, left, dyaw),
    NOT yet ÷4 / NOT yet flipped — we apply the same adapter as fastwam_agent."""
    traj = raw_model_out.astype(np.float64).copy()
    traj[:, 0:2] /= 4.0       # undo *=4
    traj[:, 1] *= -1.0         # left → right (the new adapter)
    # Add a moving_flag column = 1 so traj_utils doesn't STOP on flag check
    flag = np.ones((traj.shape[0], 1))
    traj4 = np.concatenate([traj, flag], axis=1)

    actions = fastwam_traj_to_actions(traj4,
                                      step_size=0.25,
                                      turn_angle_deg=15,
                                      lookahead=4)
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for a in actions:
        counts[a] += 1
    print(f"  {label:<55s} → "
          f"FWD={counts[1]:>2} LEFT={counts[2]:>2} RIGHT={counts[3]:>2} "
          f"STOP={counts[0]:>2} (total={len(actions)})")
    print(f"      raw seq: {[NAMES[a] for a in actions]}")


def main():
    print("=" * 80)
    print("Scenario suite: what does traj_utils do for various model outputs?")
    print("(All inputs are in TRAINING convention before adapter)")
    print("=" * 80)

    # --- S1 — model outputs essentially zero motion (likely real behavior
    #          when actually facing a wall, since training never saw rotations) ---
    print("\nS1 — Model outputs ~zero motion (likely real wall-facing behavior):")
    s1 = np.zeros((8, 3))
    s1[:, :2] = 0.001   # tiny noise in fwd & left
    run("zero motion", s1 * 4)  # *4 because training has *=4 normalization

    # --- S2 — small leftward arc (e.g. wall slightly to the right ahead) ---
    print("\nS2 — Small leftward arc (wall slightly to the right ahead):")
    s2 = np.zeros((8, 3))
    s2[:, 0] = 0.1  # forward 0.1 m per step → 0.8 m total
    s2[:, 1] = np.linspace(0, 0.2, 8)  # gradually leftward 0.2 m
    run("forward 0.1m/step + drift left 0.2m", s2 * 4)

    # --- S3 — IDEAL: backward motion (what we'd want for "facing wall, must turn 180°") ---
    print("\nS3 — IDEAL '180° turn' behavior: model outputs BACKWARD trajectory:")
    s3 = np.zeros((8, 3))
    s3[:, 0] = -0.1  # backward 0.1 m per step
    run("backward 0.1m/step (forward=-0.8m total)", s3 * 4)

    # --- S4 — IDEAL '90° left turn': model outputs purely leftward motion ---
    print("\nS4 — IDEAL '90° left turn' behavior: model outputs LEFTWARD trajectory:")
    s4 = np.zeros((8, 3))
    s4[:, 1] = 0.1  # left 0.1 m per step (training convention dim1=+left)
    run("leftward 0.1m/step (left=+0.8m total)", s4 * 4)

    # --- S5 — IDEAL '90° right turn': model outputs purely rightward motion ---
    print("\nS5 — IDEAL '90° right turn' behavior: model outputs RIGHTWARD trajectory:")
    s5 = np.zeros((8, 3))
    s5[:, 1] = -0.1  # in training conv, +left is +; rightward = negative dim1
    run("rightward 0.1m/step (right=+0.8m total)", s5 * 4)

    # --- S6 — Diagonal: forward + left (45° smooth bend) ---
    print("\nS6 — Smooth 45° bend left (forward=0.5m, left=0.5m total):")
    s6 = np.zeros((8, 3))
    s6[:, 0] = 0.0625  # 0.5 / 8
    s6[:, 1] = 0.0625  # 0.5 / 8
    run("diagonal fwd+left", s6 * 4)

    print()
    print("=" * 80)
    print("INTERPRETATION:")
    print("  S1: zero motion → traj_utils outputs only STOP / nothing → agent stuck")
    print("  S3: traj_utils CAN handle backward via large LEFT/RIGHT bursts ONLY IF")
    print("      the model outputs negative-forward, which it almost never will.")
    print("  S4/S5: traj_utils can correctly issue LEFT vs RIGHT — adapter works.")
    print("  S6: agent should produce a few FWD + a few LEFT, smoothly mixed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
