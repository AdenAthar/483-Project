"""
Antipodal grasp planner for a cylindrical pencil.

Computes a top-down gripper approach pose aligned with the pencil's long axis,
then checks whether the IK controller can converge to that pose from a given
home configuration.
"""

import numpy as np
from kinematics import forward_kinematics, Q_MIN, Q_MAX
from controller import control_step


# --------------------------------------------------------------------------- #
# Grasp pose computation
# --------------------------------------------------------------------------- #

def grasp_pose(T_pencil: np.ndarray, lift_height: float = 0.15) -> np.ndarray:
    """
    Compute a top-down antipodal grasp pose above a pencil.

    The gripper z-axis points downward; its x-axis aligns with the pencil's
    long axis so that the fingers straddle the barrel.

    Parameters
    ----------
    T_pencil    : (4,4) pencil centre pose in world frame
    lift_height : float  metres above the pencil centre for the approach

    Returns
    -------
    T_grasp : (4,4) gripper target pose in world frame
    """
    p_pencil = T_pencil[:3, 3]

    z_g = np.array([0.0, 0.0, -1.0])          # gripper z points straight down
    x_g = T_pencil[:3, 0].copy()              # align with pencil long axis
    x_g_norm = np.linalg.norm(x_g)
    if x_g_norm < 1e-9:
        x_g = np.array([1.0, 0.0, 0.0])
    else:
        x_g /= x_g_norm

    y_g = np.cross(z_g, x_g)
    y_norm = np.linalg.norm(y_g)
    if y_norm < 1e-9:                         # degenerate: z_g ∥ x_g
        x_g = np.array([0.0, 1.0, 0.0])
        y_g = np.cross(z_g, x_g)
        y_norm = np.linalg.norm(y_g)
    y_g /= y_norm

    R_grasp = np.column_stack([x_g, y_g, z_g])
    p_grasp = p_pencil + np.array([0.0, 0.0, lift_height])

    T_grasp = np.eye(4)
    T_grasp[:3, :3] = R_grasp
    T_grasp[:3, 3] = p_grasp
    return T_grasp


# --------------------------------------------------------------------------- #
# IK convergence check
# --------------------------------------------------------------------------- #

def ik_solve(q0: np.ndarray, T_target: np.ndarray,
             tol: float = 5e-3,
             max_iter: int = 800,
             dt: float = 0.01,
             kv: float = 8.0,
             kw: float = 0.5) -> tuple[bool, np.ndarray, float]:
    """
    Run the pseudoinverse controller until it converges to *T_target* or
    exhausts *max_iter* iterations.

    A low orientation gain (kw) is intentional: for the grasp-success
    experiment we primarily care about reaching the target position, not
    matching the exact gripper orientation.

    Parameters
    ----------
    q0       : (n,) starting joint configuration
    T_target : (4,4) desired end-effector pose
    tol      : float  position tolerance [m] that counts as success
    max_iter : int    maximum control steps

    Returns
    -------
    success   : bool
    q_sol     : (n,) joint angles at termination
    final_err : float  final position error [m]
    """
    q = q0.copy()
    for _ in range(max_iter):
        T_cur = forward_kinematics(q)
        err = float(np.linalg.norm(T_target[:3, 3] - T_cur[:3, 3]))
        if err < tol:
            return True, q, err
        q, _ = control_step(q, T_target, dt, kv, kw)

    T_cur = forward_kinematics(q)
    final_err = float(np.linalg.norm(T_target[:3, 3] - T_cur[:3, 3]))
    return False, q, final_err

