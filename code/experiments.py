"""
Run all four simulation experiments and print result tables.

Usage
-----
    python experiments.py

All experiments use a fixed random seed (42) for reproducibility.
"""

import numpy as np

from kinematics import forward_kinematics, N_JOINTS
from controller import track_waypoints, control_step
from grasp_planner import grasp_pose, ik_solve
from trajectory_generation import polylines_to_3d, _house_polylines

RNG = np.random.default_rng(42)
SEP = "-" * 55


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def home_config() -> np.ndarray:
    """UR5 'ready' configuration (all joints well within ±π limits)."""
    return np.array([0.0, -np.pi / 4, np.pi / 2,
                     -np.pi / 4, -np.pi / 2, 0.0])


# Workspace centre — the actual EE (x,y) the arm converges to from home
# when targeting z=0.011 with position-only control.  Determined empirically.
_WS_X = -0.45
_WS_Y = -0.21
_Z_TABLE = 0.011   # drawing surface height [m]
_Z_UP    = 0.10    # pen-up transit height [m]


# --------------------------------------------------------------------------- #
# Internal drawing-control helper (position priority, full 6-DOF arm)
# --------------------------------------------------------------------------- #

def _draw_step(q: np.ndarray, T_des,
              dt: float = 0.01, kv: float = 5.0,
              q_null=None):
    """Position-only pseudoinverse step used for all drawing experiments."""
    return control_step(q, T_des, dt=dt, kv=kv, kw=0.0,
                        lam=1e-3, q_null=q_null,
                        position_only=True)


def _pre_position(q0: np.ndarray, T_target, n_iter: int = 500,
                  dt: float = 0.01, kv: float = 5.0) -> np.ndarray:
    """Warm-start: run IK from q0 to T_target using position-only control.
    Returns the final joint configuration."""
    q = q0.copy()
    q_null = q0.copy()
    for _ in range(n_iter):
        q, _ = _draw_step(q, T_target, dt=dt, kv=kv, q_null=q_null)
    return q


def paper_pose() -> np.ndarray:
    """Fixed paper pose: flat on table, centred in the arm's reachable region."""
    T = np.eye(4)
    T[0, 3] = _WS_X
    T[1, 3] = _WS_Y
    T[2, 3] = _Z_TABLE
    return T


def pencil_pose(xy_offset: np.ndarray) -> np.ndarray:
    """Pencil resting on the table at a given (dx, dy) offset from workspace centre."""
    T = np.eye(4)
    T[0, 3] = _WS_X + xy_offset[0]
    T[1, 3] = _WS_Y + xy_offset[1]
    T[2, 3] = 0.02   # pencil sits 2 cm above table
    return T


# --------------------------------------------------------------------------- #
# Waypoint builders for geometric shapes
# --------------------------------------------------------------------------- #

def _line_waypoints(T_paper: np.ndarray, length: float = 0.10,
                    n: int = 15) -> list:
    pts = np.column_stack([
        np.linspace(-length / 2, length / 2, n),
        np.zeros(n),
    ])
    return polylines_to_3d([pts], T_paper)[0]['waypoints']


def _square_waypoints(T_paper: np.ndarray, side: float = 0.10,
                      n_side: int = 9) -> list:
    pts: list[list[float]] = []
    for t in np.linspace(0, 1, n_side, endpoint=False):
        pts.append([-side / 2 + t * side,  -side / 2])
    for t in np.linspace(0, 1, n_side, endpoint=False):
        pts.append([ side / 2,             -side / 2 + t * side])
    for t in np.linspace(0, 1, n_side, endpoint=False):
        pts.append([ side / 2 - t * side,   side / 2])
    for t in np.linspace(0, 1, n_side, endpoint=False):
        pts.append([-side / 2,              side / 2 - t * side])
    pts.append(pts[0])   # close the loop
    return polylines_to_3d([np.array(pts)], T_paper)[0]['waypoints']


def _circle_waypoints(T_paper: np.ndarray, radius: float = 0.05,
                      n: int = 36) -> list:
    angles = np.linspace(0, 2 * np.pi, n + 1)
    pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return polylines_to_3d([pts], T_paper)[0]['waypoints']


# --------------------------------------------------------------------------- #
# Experiment 1 — Grasp success rate
# --------------------------------------------------------------------------- #

def run_grasp(n_trials: int = 50,
              noise_levels: tuple = (0.0, 0.005, 0.010)) -> None:
    print(f"\n{SEP}")
    print("Experiment 1: Grasp Success Rate")
    print(SEP)
    print(f"  {'Condition':<28} {'Successes':>10}  {'Rate':>5}")
    print(f"  {'-'*28} {'-'*10}  {'-'*5}")

    q0 = home_config()
    offsets = RNG.uniform(-0.05, 0.05, size=(n_trials, 2))
    # Gripper half-opening: grasp succeeds only when the arm reaches within
    # this radius of the TRUE pencil position (models finite gripper width).
    GRIPPER_HALF = 0.008   # 8 mm  (pencil ≈ 7 mm diameter)

    for noise in noise_levels:
        successes = 0
        for i in range(n_trials):
            T_p_true  = pencil_pose(offsets[i])              # true pose
            T_p_noisy = T_p_true.copy()
            if noise > 0.0:
                T_p_noisy[:3, 3] += RNG.normal(0.0, noise, 3)
            T_g = grasp_pose(T_p_noisy)                      # approach the noisy estimate
            ok, q_sol, _ = ik_solve(q0, T_g)
            if ok:
                # Check if the arm is actually close enough to the true pencil
                ee = forward_kinematics(q_sol)[:3, 3]
                true_approach = grasp_pose(T_p_true)[:3, 3]
                if np.linalg.norm(ee - true_approach) <= GRIPPER_HALF:
                    successes += 1

        label = ("Nominal (known pose)" if noise == 0.0
                 else f"+/-{int(noise * 1000):d} mm pose noise")
        print(f"  {label:<28} {successes:>5}/{n_trials:<4}  "
              f"{successes/n_trials*100:>4.0f}%")


# --------------------------------------------------------------------------- #
# Experiment 2 — Trajectory tracking error
# --------------------------------------------------------------------------- #

def run_tracking() -> None:
    print(f"\n{SEP}")
    print("Experiment 2: Trajectory Tracking Error")
    print(SEP)
    print(f"  {'Shape':<10} {'RMS (mm)':>10}  {'Peak (mm)':>10}")
    print(f"  {'-'*10} {'-'*10}  {'-'*10}")

    q0 = home_config()
    T_paper = paper_pose()

    shapes = [
        ("Line",   _line_waypoints(T_paper)),
        ("Square", _square_waypoints(T_paper)),
        ("Circle", _circle_waypoints(T_paper)),
    ]

    for name, wps in shapes:
        # Warm-start: pre-position to the first waypoint
        q_start = _pre_position(q0, wps[0])

        # Track the full trajectory: 50 steps per waypoint, record last 20
        n_total, n_record = 50, 20
        all_errors: list[float] = []
        q = q_start.copy()
        q_null = q_start.copy()
        for T_des in wps:
            for s in range(n_total):
                q, err = _draw_step(q, T_des, q_null=q_null)
                if s >= n_total - n_record:
                    all_errors.append(err)

        errors = np.array(all_errors)
        rms  = float(np.sqrt(np.mean(errors ** 2))) * 1e3
        peak = float(np.max(errors)) * 1e3
        print(f"  {name:<10} {rms:>10.1f}  {peak:>10.1f}")


# --------------------------------------------------------------------------- #
# Experiment 3 — Drawing quality (house silhouette)
# --------------------------------------------------------------------------- #

def run_drawing() -> None:
    print(f"\n{SEP}")
    print("Experiment 3: Drawing Quality  (house silhouette)")
    print(SEP)

    q0 = home_config()
    T_paper = paper_pose()
    polylines = _house_polylines(half_side_m=0.05)
    strokes = polylines_to_3d(polylines, T_paper)

    all_errors: list[float] = []
    for stroke in strokes:
        # Warm-start to pen-up position above first stroke point
        q_start = _pre_position(q0, stroke['pen_up_T'])
        q = q_start.copy()
        q_null = q_start.copy()
        n_total, n_record = 50, 20
        for T_des in stroke['waypoints']:
            for s in range(n_total):
                q, err = _draw_step(q, T_des, q_null=q_null)
                if s >= n_total - n_record:
                    all_errors.append(err * 1e3)

    arr = np.array(all_errors)
    print(f"  Mean directed distance : {np.mean(arr):.1f} mm")
    print(f"  95th-percentile        : {np.percentile(arr, 95):.1f} mm")


# --------------------------------------------------------------------------- #
# Experiment 4 — Nullspace projection benefit
# --------------------------------------------------------------------------- #

def run_nullspace() -> None:
    print(f"\n{SEP}")
    print("Experiment 4: Nullspace Projection Benefit")
    print(SEP)
    print(f"  {'Condition':<30} {'Sat. steps':>12}  {'Sat. rate':>10}")
    print(f"  {'-'*30} {'-'*12}  {'-'*10}")

    from kinematics import Q_MIN, Q_MAX

    q0 = home_config()
    T_paper = paper_pose()
    # Run 4 laps of a combined shape to build up enough motion
    base_wps = _square_waypoints(T_paper) + _circle_waypoints(T_paper)
    wps = base_wps * 4

    # Warm-start to first waypoint
    q_start = _pre_position(q0, wps[0])

    for use_ns in [False, True]:
        q = q_start.copy()
        q_null = q_start.copy() if use_ns else None
        sat = 0
        total = 0
        for T_des in wps:
            for _ in range(30):
                q, _ = _draw_step(q, T_des, q_null=q_null)
                total += 1
                if np.any(q <= Q_MIN + 1e-3) or np.any(q >= Q_MAX - 1e-3):
                    sat += 1
        label = "With nullspace" if use_ns else "Without nullspace"
        print(f"  {label:<30} {sat:>12d}  {sat/total*100:>9.0f}%")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("=" * 55)
    print("Robot Pencil Drawing — Simulation Experiments")
    print("CPS_S MATH 453 Final Project")
    print("=" * 55)

    run_grasp()
    run_tracking()
    run_drawing()
    run_nullspace()

    print(f"\n{'=' * 55}")
    print("All experiments complete.")
    print("=" * 55)

