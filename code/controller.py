"""
Jacobian pseudoinverse controller with optional nullspace projection.

Control law
-----------
    q_dot = J+(q) xi_ref  +  (I - J+(q) J(q)) q_null_dot

where
    J+(q) = J(q)^T  (J(q) J(q)^T + lambda I)^{-1}   (damped pseudoinverse)
    xi_ref = [kw * e_omega;  kv * e_v]                (reference twist)
    (I - J+ J) q_null_dot                             (nullspace term)
"""

import numpy as np
from kinematics import forward_kinematics, jacobian, Q_MIN, Q_MAX


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _rotation_error(R_des: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """
    Axis-angle orientation error as a 3-vector (world frame).

    Returns the vector  angle * axis  where R_err = R_des @ R_cur^T.
    """
    R_err = R_des @ R_cur.T
    trace_val = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(trace_val))

    if angle < 1e-9:
        return np.zeros(3)

    sin_a = np.sin(angle)
    if abs(sin_a) < 1e-9:
        # angle ≈ π — degenerate: recover axis from diagonal
        diag = (np.diag(R_err) + 1.0) / 2.0
        axis = np.sqrt(np.maximum(diag, 0.0))
        axis /= np.linalg.norm(axis) + 1e-12
        return np.pi * axis

    axis = np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ]) / (2.0 * sin_a)
    return angle * axis


# --------------------------------------------------------------------------- #
# Single control step
# --------------------------------------------------------------------------- #

def control_step(q: np.ndarray, T_des: np.ndarray,
                 dt: float = 0.01, kv: float = 5.0, kw: float = 2.0,
                 lam: float = 1e-4,
                 q_null: np.ndarray | None = None,
                 position_only: bool = False) -> tuple[np.ndarray, float]:
    """
    One Jacobian-pseudoinverse control step.

    Parameters
    ----------
    q             : (n,)  current joint configuration [rad]
    T_des         : (4,4) desired end-effector SE(3) pose
    dt            : float timestep [s]
    kv            : float translational gain  [1/s]
    kw            : float rotational gain     [1/s]
    lam           : float Tikhonov damping for the pseudoinverse
    q_null        : (n,) or None  secondary objective for nullspace projection
    position_only : bool  if True use only the 3-row position Jacobian
                    (ignores orientation error; avoids orientation ill-conditioning)

    Returns
    -------
    q_new   : (n,) updated joint angles, clamped to [Q_MIN, Q_MAX]
    pos_err : float  Cartesian position error magnitude [m]
    """
    T_cur = forward_kinematics(q)
    ev = T_des[:3, 3] - T_cur[:3, 3]
    pos_err = float(np.linalg.norm(ev))

    J = jacobian(q)
    n = J.shape[1]

    if position_only:
        # Use only the 3×n linear-velocity rows; 3-DOF nullspace is free
        Jp = J[3:, :]                                          # (3, n)
        Jp_pinv = Jp.T @ np.linalg.inv(Jp @ Jp.T + lam * np.eye(3))
        q_dot = Jp_pinv @ (kv * ev)
        if q_null is not None:
            N = np.eye(n) - Jp_pinv @ Jp
            q_dot = q_dot + N @ (q_null - q)
    else:
        ew = _rotation_error(T_des[:3, :3], T_cur[:3, :3])
        xi_ref = np.concatenate([kw * ew, kv * ev])
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(6))
        q_dot = J_pinv @ xi_ref
        if q_null is not None:
            N = np.eye(n) - J_pinv @ J
            q_dot = q_dot + N @ (q_null - q)

    q_new = np.clip(q + q_dot * dt, Q_MIN, Q_MAX)
    return q_new, pos_err


# --------------------------------------------------------------------------- #
# Waypoint tracker
# --------------------------------------------------------------------------- #

def track_waypoints(q0: np.ndarray,
                    waypoints: list,
                    dt: float = 0.01,
                    kv: float = 5.0,
                    kw: float = 2.0,
                    steps_per_wp: int = 40,
                    lam: float = 1e-4,
                    use_nullspace: bool = True) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Track a sequence of SE(3) waypoints using the pseudoinverse controller.

    Parameters
    ----------
    q0            : (n,) initial joint configuration
    waypoints     : list of (4,4) SE(3) target transforms
    steps_per_wp  : int   control steps spent per waypoint
    use_nullspace : bool  enable nullspace projection toward q0

    Returns
    -------
    errors    : (N,) position error at every control step [m]
                where N = len(waypoints) * steps_per_wp
    q_final   : (n,) final joint configuration
    sat_count : int  number of steps where at least one joint limit is active
    """
    q = q0.copy()
    q_null = q0.copy() if use_nullspace else None
    errors: list[float] = []
    sat_count = 0

    for T_des in waypoints:
        for _ in range(steps_per_wp):
            q, err = control_step(q, T_des, dt, kv, kw, lam, q_null)
            errors.append(err)
            if np.any(q <= Q_MIN + 1e-3) or np.any(q >= Q_MAX - 1e-3):
                sat_count += 1

    return np.array(errors), q, sat_count

