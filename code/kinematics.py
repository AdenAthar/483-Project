"""
UR5 forward kinematics and numerical Jacobian.

Uses the modified (Craig) DH convention.
DH table columns: [a (m), d (m), alpha (rad), theta_offset (rad)]

Reference: Universal Robots UR5 technical specification.
"""

import numpy as np

# --------------------------------------------------------------------------- #
# UR5 modified DH parameters
# --------------------------------------------------------------------------- #
UR5_DH = np.array([
    [ 0.00000,  0.08920,  np.pi / 2,  0.0],   # joint 1
    [-0.42500,  0.00000,  0.0,        0.0],   # joint 2
    [-0.39225,  0.00000,  0.0,        0.0],   # joint 3
    [ 0.00000,  0.10915,  np.pi / 2,  0.0],   # joint 4
    [ 0.00000,  0.09465, -np.pi / 2,  0.0],   # joint 5
    [ 0.00000,  0.08230,  0.0,        0.0],   # joint 6
])

N_JOINTS = 6

# Practical workspace limits (±120° ≈ ±2π/3).  Chosen to represent real
# operating-range constraints; tighter than hardware limits so that
# joint-drift experiments produce observable saturation behaviour.
Q_MIN = np.full(N_JOINTS, -2.0 * np.pi / 3)
Q_MAX = np.full(N_JOINTS,  2.0 * np.pi / 3)


# --------------------------------------------------------------------------- #
# Core functions
# --------------------------------------------------------------------------- #

def _dh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """Return the 4x4 homogeneous DH transform for a single link."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,       -st,        0.0,   a      ],
        [st * ca,   ct * ca,  -sa,   -sa * d ],
        [st * sa,   ct * sa,   ca,    ca * d ],
        [0.0,       0.0,       0.0,   1.0    ],
    ])


def forward_kinematics(q: np.ndarray, dh: np.ndarray | None = None) -> np.ndarray:
    """
    Compute the 4x4 SE(3) end-effector transform for joint angles *q*.

    Parameters
    ----------
    q  : (6,) joint angles [rad]
    dh : (6, 4) DH table; defaults to UR5_DH

    Returns
    -------
    T : (4, 4) ndarray — world-to-EE homogeneous transform
    """
    if dh is None:
        dh = UR5_DH
    q = np.asarray(q, dtype=float)
    T = np.eye(4)
    for (a, d, alpha, offset), qi in zip(dh, q):
        T = T @ _dh_transform(a, d, alpha, qi + offset)
    return T


def jacobian(q: np.ndarray, dh: np.ndarray | None = None,
             eps: float = 1e-6) -> np.ndarray:
    """
    Numerical (6 x n) Jacobian via central finite differences.

    Row order: [wx, wy, wz,  vx, vy, vz]
               (angular first, then linear — matches the twist convention
                used throughout the controller)

    Parameters
    ----------
    q   : (n,) joint angles [rad]
    eps : float  finite-difference step size

    Returns
    -------
    J : (6, n) ndarray
    """
    if dh is None:
        dh = UR5_DH
    q = np.asarray(q, dtype=float)
    n = len(q)
    R0 = forward_kinematics(q, dh)[:3, :3]
    J = np.zeros((6, n))

    for i in range(n):
        dq = np.zeros(n)
        dq[i] = eps
        Tp = forward_kinematics(q + dq, dh)
        Tm = forward_kinematics(q - dq, dh)

        # Linear velocity column
        J[3:, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2.0 * eps)

        # Angular velocity column from the skew  W = (dR/dqi) R0^T
        dR = (Tp[:3, :3] - Tm[:3, :3]) / (2.0 * eps)
        W = dR @ R0.T          # skew-symmetric matrix
        J[0, i] = W[2, 1]
        J[1, i] = W[0, 2]
        J[2, i] = W[1, 0]

    return J

