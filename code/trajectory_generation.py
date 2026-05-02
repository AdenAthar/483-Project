"""
2-D image to 3-D pen-tip waypoint conversion.

Pipeline
--------
  bitmap → Canny edge detection → polylines (paper frame, metres)
         → 3-D SE(3) waypoints in world frame   (pen-up / pen-down)

If opencv-python is not installed the module falls back to a hardcoded
house-silhouette used for testing.
"""

import numpy as np

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# --------------------------------------------------------------------------- #
# Edge extraction
# --------------------------------------------------------------------------- #

def image_to_polylines(image_path: str,
                       scale_m_per_px: float = 5e-4,
                       max_contours: int = 20,
                       canny_low: int = 50,
                       canny_high: int = 150) -> list[np.ndarray]:
    """
    Run Canny edge detection on a bitmap and return a list of polylines.

    Each polyline is an (M, 2) ndarray of (x, y) coordinates in metres,
    centred at the image centre.

    Parameters
    ----------
    image_path     : str   path to input bitmap
    scale_m_per_px : float metres per pixel  (default ≈ 0.5 mm/px)
    max_contours   : int   keep only the longest *max_contours* contours

    Returns
    -------
    polylines : list of (M_k, 2) ndarrays
    """
    if _HAS_CV2:
        img = _cv2.imread(str(image_path), _cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return _canny_polylines(img, scale_m_per_px, max_contours,
                                    canny_low, canny_high)
    # Fallback
    return _house_polylines()


def _canny_polylines(img: np.ndarray, scale: float,
                     max_contours: int, low: int, high: int) -> list[np.ndarray]:
    edges = _cv2.Canny(img, low, high)
    contours, _ = _cv2.findContours(edges, _cv2.RETR_LIST,
                                    _cv2.CHAIN_APPROX_SIMPLE)
    # Sort longest-first and trim
    contours = sorted(contours, key=len, reverse=True)[:max_contours]
    h, w = img.shape
    cx, cy = w / 2.0, h / 2.0
    result = []
    for c in contours:
        pts = c[:, 0, :].astype(float)       # (M, 2)  pixel coordinates
        pts = (pts - np.array([cx, cy])) * scale
        result.append(pts)
    return result


def _house_polylines(half_side_m: float = 0.05) -> list[np.ndarray]:
    """House silhouette used as a built-in test / fallback target."""
    s = half_side_m
    walls = np.array([[-s, -s], [ s, -s], [ s,  s], [-s,  s], [-s, -s]])
    roof  = np.array([[-s,  s], [0.0, 2 * s], [ s,  s]])
    return [walls, roof]


# --------------------------------------------------------------------------- #
# 2-D → 3-D lift
# --------------------------------------------------------------------------- #

def polylines_to_3d(polylines: list[np.ndarray],
                    T_paper: np.ndarray,
                    z_down: float = 0.001,
                    z_up: float = 0.05) -> list[dict]:
    """
    Convert 2-D polylines on a paper frame to a list of drawing strokes.

    Each stroke dictionary contains:
      'pen_up_T'  : (4,4) SE(3)  pen-up transit pose above the stroke start
      'waypoints' : list of (4,4) SE(3)  pen-down waypoint poses

    Parameters
    ----------
    polylines : list of (M_k, 2) ndarrays  — (x, y) in paper frame [m]
    T_paper   : (4,4) SE(3)  paper frame pose in world frame
                (origin = paper centre; x/y in paper plane; z points up)
    z_down    : float  pen-tip height above paper surface when drawing [m]
    z_up      : float  pen-tip height for transit moves [m]

    Returns
    -------
    strokes : list of dicts  (one per polyline)
    """
    R_paper = T_paper[:3, :3]
    p_paper = T_paper[:3, 3]

    def _tip_pose(x: float, y: float, z: float) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = R_paper
        T[:3, 3] = p_paper + R_paper @ np.array([x, y, z])
        return T

    strokes = []
    for poly in polylines:
        wps = [_tip_pose(float(x), float(y), z_down) for x, y in poly]
        t_up = _tip_pose(float(poly[0, 0]), float(poly[0, 1]), z_up)
        strokes.append({'pen_up_T': t_up, 'waypoints': wps})
    return strokes

