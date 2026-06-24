"""
Diagnostic script to analyze training action label distribution.
Checks whether theta (d_theta / heading) values in training labels are systematically biased.

Usage:
    python scripts/diagnose_action_labels.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ─── copy of core transforms from nav_video_dataset.py ───────────────────────

def get_trajectory_relative_to_frame(extrinsics: np.ndarray, camera_deg: float = 0) -> np.ndarray:
    T_camera2robot = np.array(
        [[[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
    )
    T_robot2camera = np.array(
        [[[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
    )

    if camera_deg is not None and camera_deg != 0:
        camera_rad = np.radians(camera_deg)
        T_deg = np.array(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, np.cos(-camera_rad), -np.sin(-camera_rad), 0.0],
                    [0.0, np.sin(-camera_rad), np.cos(-camera_rad), 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            dtype=np.float32,
        )
        T_robot2camera = np.matmul(T_robot2camera, T_deg)
        T_camera2robot = np.linalg.inv(T_robot2camera)

    extrinsics_robot = np.matmul(extrinsics, T_camera2robot)
    T_ref = extrinsics_robot[0]
    T_ref_inv = np.linalg.inv(T_ref)
    relative_to_ref = np.matmul(T_ref_inv[np.newaxis, :, :], extrinsics_robot)
    relative_translations = relative_to_ref[:, :2, 3]
    relative_yaws = np.arctan2(relative_to_ref[:, 1, 0], relative_to_ref[:, 0, 0])
    return np.concatenate((relative_translations, relative_yaws.reshape(-1, 1)), axis=-1)


def smooth_and_resample_trajectory(points, sample_length=33, interval=0.1):
    total_distance = sample_length * interval
    if len(points) == 0:
        return np.zeros((sample_length, 2))
    if len(points) == 1:
        return np.tile(points[0], (sample_length, 1))
    diff = np.diff(points, axis=0)
    segment_lengths = np.sqrt(np.sum(diff**2, axis=1))
    cumulative_distances = np.cumsum(segment_lengths)
    cumulative_distances = np.insert(cumulative_distances, 0, 0)
    if len(points) > 3:
        cs_x = CubicSpline(cumulative_distances, points[:, 0])
        cs_y = CubicSpline(cumulative_distances, points[:, 1])
        dense_distances = np.linspace(0, cumulative_distances[-1], max(50, len(points) * 2))
        x_smooth = cs_x(dense_distances)
        y_smooth = cs_y(dense_distances)
        smoothed_points = np.column_stack((x_smooth, y_smooth))
        smooth_diff = np.diff(smoothed_points, axis=0)
        smooth_segment_lengths = np.sqrt(np.sum(smooth_diff**2, axis=1))
        smooth_cumulative_distances = np.cumsum(smooth_segment_lengths)
        smooth_cumulative_distances = np.insert(smooth_cumulative_distances, 0, 0)
    else:
        smoothed_points = points
        smooth_cumulative_distances = cumulative_distances
    target_distances = np.linspace(0, total_distance, sample_length)
    resampled = np.zeros((sample_length, 2))
    for i, target_dist in enumerate(target_distances):
        if target_dist >= smooth_cumulative_distances[-1]:
            resampled[i] = smoothed_points[-1]
            continue
        segment_idx = np.searchsorted(smooth_cumulative_distances, target_dist, side='right') - 1
        start_dist = smooth_cumulative_distances[segment_idx]
        end_dist = smooth_cumulative_distances[segment_idx + 1]
        t = (target_dist - start_dist) / (end_dist - start_dist + 1e-8)
        resampled[i] = smoothed_points[segment_idx] + t * (smoothed_points[segment_idx + 1] - smoothed_points[segment_idx])
    return resampled


def xy_to_delta_xyt(xy_actions):
    vectors = np.diff(xy_actions, axis=0)
    yaw = np.arctan2(vectors[:, 1], vectors[:, 0])
    delta_yaw = np.diff(yaw)
    delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi
    delta_yaw = np.concatenate([[yaw[0]], delta_yaw])
    return np.concatenate([vectors, delta_yaw[:, None]], axis=1)


def interpolate_and_resample_trajectory(absolute_trajectories, predict_step_num=8):
    start_point = np.array([[0.0, 0.0]])
    traj = absolute_trajectories[..., :2]
    steps = traj[1:] - traj[:-1]
    steps_sq = (steps**2).sum(axis=-1)
    mask = steps_sq > 0.05
    filtered_traj = traj[1:][mask]
    filtered_traj = np.concatenate([start_point, filtered_traj], axis=0)
    resampled_trajectories = smooth_and_resample_trajectory(filtered_traj, sample_length=predict_step_num + 1)
    resampled_relative_poses = xy_to_delta_xyt(resampled_trajectories)
    resampled_relative_poses[:, 0:2] *= 4  # normalization factor
    return resampled_trajectories, resampled_relative_poses


# ─── Test the coordinate transform directly ──────────────────────────────────

def test_identity_transform():
    """
    Test: if agent moves forward (along camera forward direction), what direction does the action point?

    In the overhead camera (125cm_30deg), the camera looks roughly downward at 30 degrees.
    "Forward" in real world should map to positive dx in action space (theta ~ 0).
    """
    print("\n" + "="*70)
    print("TEST 1: Sanity check – forward motion → theta should be ~0")
    print("="*70)

    # Simulate: agent walks forward along +Z in world coords
    # Overhead camera is pitched down 30 degrees from horizontal
    # Camera extrinsic: camera looks along +Z world axis (simplified)

    # Create a simple scenario: agent at (0,0,0), (1,0,0), (2,0,0), (3,0,0) in world
    # Camera extrinsic = world-to-camera transform
    # For a camera at height 1.25m looking forward at 30deg pitch:

    # World: X=right, Y=up, Z=forward (standard robot convention)
    # Camera at height 1.25m, pitched 30 deg downward:
    #   Camera X = World X
    #   Camera Y = World Z (scene depth)   (after -30 deg pitch)
    #   Camera Z = -World Y (up)

    # Let's use a concrete test: identity camera (no rotation) vs pitched camera
    N = 5
    world_positions = np.array([[i, 0.0, 0.0, 1.0] for i in range(N)])  # agent moves along X

    # Extrinsic = camera-to-world (as used in VLN datasets typically)
    # For simplicity, assume camera frame = world frame (identity rotation)
    extrinsics_identity = np.tile(np.eye(4), (N, 1, 1)).astype(np.float32)
    for i in range(N):
        extrinsics_identity[i, 0, 3] = float(i)  # agent moves along X

    result_0deg = get_trajectory_relative_to_frame(extrinsics_identity, camera_deg=0)
    result_30deg = get_trajectory_relative_to_frame(extrinsics_identity, camera_deg=30)

    print(f"\nIdentity camera, agent moves along +X:")
    print(f"  camera_deg=0:  relative_xy = {result_0deg[:, :2]}")
    print(f"  camera_deg=30: relative_xy = {result_30deg[:, :2]}")

    _, actions_0deg = interpolate_and_resample_trajectory(result_0deg, 8)
    _, actions_30deg = interpolate_and_resample_trajectory(result_30deg, 8)

    print(f"\n  camera_deg=0  actions (dx,dy,dtheta):\n    {actions_0deg[:3]}")
    print(f"  camera_deg=30 actions (dx,dy,dtheta):\n    {actions_30deg[:3]}")

    # Expected: theta should be close to 0 if "forward" maps correctly
    print(f"\n  camera_deg=0  theta[0] = {np.degrees(actions_0deg[0, 2]):.1f} deg (expect ~0)")
    print(f"  camera_deg=30 theta[0] = {np.degrees(actions_30deg[0, 2]):.1f} deg (expect ~0)")


def test_camera_rotation_matrix():
    """
    Test the T_camera2robot rotation matrix directly.
    Shows what happens to a forward-moving trajectory.
    """
    print("\n" + "="*70)
    print("TEST 2: T_camera2robot matrix analysis")
    print("="*70)

    T_camera2robot = np.array(
        [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    print(f"\nT_camera2robot:\n{T_camera2robot}")

    # Camera axes in robot frame:
    # Camera +X → where does it go in robot?
    cam_x = T_camera2robot[:3, 0]  # column 0
    cam_y = T_camera2robot[:3, 1]  # column 1
    cam_z = T_camera2robot[:3, 2]  # column 2
    print(f"\nCamera +X axis in robot frame: {cam_x}")
    print(f"Camera +Y axis in robot frame: {cam_y}")
    print(f"Camera +Z axis in robot frame: {cam_z}")

    print("\nInterpretation:")
    print(f"  Camera +X → robot {cam_x} (pointing {'right' if cam_x[0]>0 else 'left' if cam_x[0]<0 else 'up' if cam_x[1]>0 else 'down' if cam_x[1]<0 else 'forward' if cam_x[2]>0 else 'backward'})")

    # Check: what does the relative_to_ref[:, 0, 0] and [:, 1, 0] measure for yaw?
    # relative_yaws = arctan2(relative_to_ref[:, 1, 0], relative_to_ref[:, 0, 0])
    # These are the x and y components of the robot's forward direction in the reference frame
    print("\nYaw extraction: arctan2(R[1,0], R[0,0])")
    print("  R[0,0] = x component of robot's X-axis in ref frame")
    print("  R[1,0] = y component of robot's X-axis in ref frame")

    # For identity rotation (no rotation), R[0,0]=1, R[1,0]=0 → yaw=0
    # Forward rotation (90 deg CCW), R[0,0]=0, R[1,0]=1 → yaw=90 deg


def analyze_real_episode(parquet_path, camera_key='125cm_30deg', camera_deg=30, predict_step_num=8):
    """Analyze a real episode from the training data."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {parquet_path}")
    print(f"  camera_key={camera_key}, camera_deg={camera_deg}")
    print(f"{'='*70}")

    df = pd.read_parquet(parquet_path, columns=[f"pose.{camera_key}"])
    poses_raw = df[f"pose.{camera_key}"].tolist()
    poses = np.array([np.vstack(p) for p in poses_raw])
    print(f"  Episode length: {len(poses)} frames")
    print(f"\n  First pose matrix:\n{poses[0]}")
    print(f"\n  Second pose matrix:\n{poses[1]}")

    # Compute trajectory relative to frame 0
    traj = get_trajectory_relative_to_frame(poses, camera_deg=camera_deg)
    print(f"\n  Trajectory (x, y, yaw) first 5 frames:")
    for i in range(min(5, len(traj))):
        print(f"    frame {i}: x={traj[i,0]:.4f}, y={traj[i,1]:.4f}, yaw={np.degrees(traj[i,2]):.2f} deg")

    # Compute actions for a segment
    segment = min(17, len(poses))
    discrete_traj = get_trajectory_relative_to_frame(poses[:segment], camera_deg=camera_deg)
    resampled_traj, resampled_actions = interpolate_and_resample_trajectory(discrete_traj, predict_step_num)

    print(f"\n  Resampled trajectory (x,y) - pre-normalization ({segment} frames → {predict_step_num} steps):")
    for i in range(len(resampled_traj)):
        print(f"    step {i}: x={resampled_traj[i,0]:.4f}, y={resampled_traj[i,1]:.4f}")

    print(f"\n  Action labels (dx, dy, dtheta) [after *4 normalization]:")
    for i in range(predict_step_num):
        dx, dy, dth = resampled_actions[i]
        r = np.sqrt(dx**2 + dy**2)
        th = np.degrees(np.arctan2(dy, dx))
        print(f"    step {i}: dx={dx:.4f}, dy={dy:.4f}, dtheta={np.degrees(dth):.2f} deg | r={r:.4f}, theta={th:.2f} deg")

    return resampled_actions, traj


def analyze_action_distribution(dataset_dir, n_episodes=50, camera_key='125cm_30deg', camera_deg=30, predict_step_num=8):
    """Sample n_episodes and collect action label statistics."""
    print(f"\n{'='*70}")
    print(f"Statistical analysis: {n_episodes} episodes from {dataset_dir}")
    print(f"{'='*70}")

    import glob
    parquet_files = sorted(glob.glob(os.path.join(dataset_dir, '*/data/chunk-000/*.parquet')))
    if not parquet_files:
        print(f"  No parquet files found!")
        return None

    print(f"  Found {len(parquet_files)} episodes total")
    np.random.seed(42)
    selected = np.random.choice(parquet_files, min(n_episodes, len(parquet_files)), replace=False)

    all_dx, all_dy, all_dtheta = [], [], []
    all_r, all_theta = [], []
    moving_count = 0
    total_steps = 0

    for pf in selected:
        try:
            df = pd.read_parquet(pf, columns=[f"pose.{camera_key}"])
            poses_raw = df[f"pose.{camera_key}"].tolist()
            poses = np.array([np.vstack(p) for p in poses_raw])

            # Sample from various positions in the episode
            for start in range(0, len(poses) - 2, 8):
                end = min(start + 17, len(poses))
                discrete_traj = get_trajectory_relative_to_frame(poses[start:end], camera_deg=camera_deg)
                _, actions = interpolate_and_resample_trajectory(discrete_traj, predict_step_num)

                # Check moving
                traj_xy = discrete_traj[:, :2]
                steps = traj_xy[1:] - traj_xy[:-1]
                n_moving = (np.sum(steps**2, axis=1) > 0.05).sum()
                if n_moving < 2:
                    continue  # skip stopped segments

                for step_actions in actions:
                    dx, dy, dth = step_actions
                    r = np.sqrt(dx**2 + dy**2)
                    th = np.degrees(np.arctan2(dy, dx))
                    all_dx.append(dx)
                    all_dy.append(dy)
                    all_dtheta.append(np.degrees(dth))
                    all_r.append(r)
                    all_theta.append(th)
                    if r > 0.01:  # effectively moving
                        moving_count += 1
                    total_steps += 1
        except Exception as e:
            pass

    if not all_r:
        print("  No data collected!")
        return None

    all_r = np.array(all_r)
    all_theta = np.array(all_theta)
    all_dtheta = np.array(all_dtheta)
    all_dx = np.array(all_dx)
    all_dy = np.array(all_dy)

    print(f"\n  Collected {total_steps} action steps ({moving_count} effectively moving)")
    print(f"\n  --- ACTION THETA (heading direction = arctan2(dy, dx)) ---")
    print(f"  mean:   {np.mean(all_theta):.2f} deg")
    print(f"  median: {np.median(all_theta):.2f} deg")
    print(f"  std:    {np.std(all_theta):.2f} deg")
    print(f"  min:    {np.min(all_theta):.2f} deg")
    print(f"  max:    {np.max(all_theta):.2f} deg")

    # Histogram of theta
    hist, edges = np.histogram(all_theta, bins=36, range=(-180, 180))
    print(f"\n  Theta distribution (36 bins, -180 to 180):")
    for i in range(0, 36, 4):
        bar_len = int(hist[i] / max(hist) * 30)
        print(f"    {edges[i]:+.0f}° to {edges[i+1]:+.0f}°: {'#'*bar_len} ({hist[i]})")

    print(f"\n  --- R (action magnitude = sqrt(dx²+dy²)) ---")
    print(f"  mean:   {np.mean(all_r):.4f}")
    print(f"  median: {np.median(all_r):.4f}")
    print(f"  std:    {np.std(all_r):.4f}")

    print(f"\n  --- DELTA THETA (turn angle per step) ---")
    print(f"  mean:   {np.mean(all_dtheta):.2f} deg")
    print(f"  median: {np.median(all_dtheta):.2f} deg")
    print(f"  std:    {np.std(all_dtheta):.2f} deg")

    print(f"\n  --- dx distribution ---")
    print(f"  mean:   {np.mean(all_dx):.4f}")
    print(f"  median: {np.median(all_dx):.4f}")
    print(f"  std:    {np.std(all_dx):.4f}")

    print(f"\n  --- dy distribution ---")
    print(f"  mean:   {np.mean(all_dy):.4f}")
    print(f"  median: {np.median(all_dy):.4f}")
    print(f"  std:    {np.std(all_dy):.4f}")

    # Check for suspicious concentration around -75 degrees
    near_minus75 = np.sum(np.abs(all_theta - (-75)) < 15) / len(all_theta) * 100
    print(f"\n  % of steps with theta in [-90, -60] (around -75°): {near_minus75:.1f}%")

    return {
        'theta': all_theta,
        'r': all_r,
        'dtheta': all_dtheta,
        'dx': all_dx,
        'dy': all_dy,
    }


def diagnose_transform_with_camera_deg(parquet_path, camera_key='125cm_30deg'):
    """
    Compare camera_deg=0 vs camera_deg=30 for same episode.
    """
    print(f"\n{'='*70}")
    print(f"Comparing camera_deg=0 vs camera_deg=30")
    print(f"{'='*70}")

    df = pd.read_parquet(parquet_path, columns=[f"pose.{camera_key}"])
    poses_raw = df[f"pose.{camera_key}"].tolist()
    poses = np.array([np.vstack(p) for p in poses_raw])[:17]

    for deg in [0, 30]:
        traj = get_trajectory_relative_to_frame(poses, camera_deg=deg)
        _, actions = interpolate_and_resample_trajectory(traj, 8)
        thetas = [np.degrees(np.arctan2(actions[i,1], actions[i,0])) for i in range(8)]
        rs = [np.sqrt(actions[i,0]**2 + actions[i,1]**2) for i in range(8)]
        print(f"\n  camera_deg={deg}:")
        print(f"    thetas: {[f'{t:.1f}' for t in thetas]}")
        print(f"    r vals: {[f'{r:.3f}' for r in rs]}")
        print(f"    mean theta: {np.mean(thetas):.1f} deg")


def check_pose_column_format(parquet_path, camera_key='125cm_30deg'):
    """Check what the raw pose data looks like and its conventions."""
    print(f"\n{'='*70}")
    print(f"Raw pose data format check")
    print(f"{'='*70}")

    df = pd.read_parquet(parquet_path)
    print(f"  Available columns: {[c for c in df.columns if 'pose' in c]}")

    col = f"pose.{camera_key}"
    if col not in df.columns:
        print(f"  Column {col} not found!")
        return

    pose0 = df[col].iloc[0]
    pose0_mat = np.vstack(pose0)

    print(f"\n  pose.{camera_key}[frame 0]:\n{pose0_mat}")
    print(f"\n  Rotation part R:\n{pose0_mat[:3,:3]}")
    print(f"  Translation t: {pose0_mat[:3,3]}")

    # Check determinant (should be ~1 for rotation matrix)
    det = np.linalg.det(pose0_mat[:3,:3])
    print(f"\n  det(R) = {det:.6f} (should be ~1)")

    # Check multiple consecutive frames
    print(f"\n  Translation changes (first 5 frames):")
    for i in range(min(5, len(df))):
        pose_i = np.vstack(df[col].iloc[i])
        t_i = pose_i[:3, 3]
        if i > 0:
            dt = t_i - t_prev
            dist = np.linalg.norm(dt)
            print(f"    frame {i}: t={t_i[:3]}, delta_t={dt[:3]}, dist={dist:.4f}m")
        else:
            print(f"    frame {i}: t={t_i[:3]}")
        t_prev = t_i.copy()

    # What does "forward" look like in camera space?
    print(f"\n  Camera Z axis (typically forward): {pose0_mat[:3,2]}")
    print(f"  Camera X axis: {pose0_mat[:3,0]}")
    print(f"  Camera Y axis: {pose0_mat[:3,1]}")


if __name__ == '__main__':
    # ── Test 1: Sanity check with known motion ──
    test_identity_transform()
    test_camera_rotation_matrix()

    # ── Test 2: Inspect real data ──
    dataset_dir = '/apdcephfs_gy6/share_303214315/jishengpeng/vlndata/InternData-N1/vln_ce/traj_data/r2r'
    first_scene = sorted(os.listdir(dataset_dir))[0]
    first_parquet = os.path.join(dataset_dir, first_scene, 'data', 'chunk-000', 'episode_000000.parquet')

    print(f"\nUsing: {first_parquet}")
    check_pose_column_format(first_parquet)
    diagnose_transform_with_camera_deg(first_parquet)

    _, _ = analyze_real_episode(first_parquet, predict_step_num=8)

    # ── Test 3: Statistical analysis ──
    stats = analyze_action_distribution(dataset_dir, n_episodes=100, predict_step_num=8)

    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    if stats is not None:
        mean_theta = np.mean(stats['theta'])
        print(f"\nMean action theta: {mean_theta:.2f} deg")
        print(f"Std action theta: {np.std(stats['theta']):.2f} deg")

        if abs(mean_theta) > 30:
            print(f"\n⚠️  WARNING: Mean theta is {mean_theta:.2f} deg — significant directional bias!")
            print("   This likely means the coordinate transform is mapping 'forward' to a non-zero angle.")
        else:
            print(f"\n✓ Mean theta is close to 0 ({mean_theta:.2f} deg) — no systematic bias in training labels.")
            print("  The -75° prediction at inference may be a different issue (e.g., wrong denormalization).")

        mean_r = np.mean(stats['r'])
        print(f"\nMean action r (magnitude): {mean_r:.4f}")
        print(f"  (training r is *4 normalized; actual trajectory step ~0.1m → expect r ~0.4)")
        if abs(mean_r - 0.4) < 0.15:
            print(f"  ✓ r is consistent with *4 normalization (0.1m * 4 = 0.4)")
        else:
            print(f"  ⚠️  r={mean_r:.4f} unexpected (expect ~0.4 for 0.1m steps)")
