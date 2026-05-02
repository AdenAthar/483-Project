# Robot Pencil Grasping and Image Drawing — Simulation Code

**CPS_S MATH 453 Final Project** — Abdur Islam, Aden Athar

## Overview

Pure-Python kinematic simulation that implements and validates every claim in `final_project.tex`. No physics engine or robot hardware is required — the simulation uses the analytical UR5 kinematic model with a numerical Jacobian computed via central finite differences.

## File Layout

| File | Description |
|---|---|
| `kinematics.py` | UR5 DH parameters, forward kinematics, Jacobian |
| `controller.py` | Jacobian pseudoinverse controller + nullspace |
| `grasp_planner.py` | Antipodal grasp pose + IK convergence check |
| `trajectory_generation.py` | Bitmap → Canny edges → polylines → SE(3) waypoints |
| `experiments.py` | All four experiments (entry point) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## Requirements

Python 3.10 or later.

- **Required:** `numpy >= 1.24`
- **Optional** (needed only for Experiment 3 with a real bitmap): `opencv-python >= 4.8`

## Installation

**Option A — pip**
```bash
pip install -r requirements.txt
```

**Option B — uv**
```bash
uv pip install -r requirements.txt
```

**Option C — conda**
```bash
conda install numpy
conda install -c conda-forge opencv
```

## Running the Experiments

From inside the `code/` folder:

```bash
python experiments.py
```

The script runs all four experiments in sequence and prints a result table for each one. With a fixed random seed (42) the output is fully reproducible across runs.

### Expected Output

**Experiment 1: Grasp Success Rate**

| Condition | Successes | Rate |
|---|---|---|
| Nominal (known pose) | 50/50 | 100% |
| ±5 mm pose noise | 16/50 | 32% |
| ±10 mm pose noise | 9/50 | 18% |

**Experiment 2: Trajectory Tracking Error**

| Shape | RMS (mm) | Peak (mm) |
|---|---|---|
| Line | 1.3 | 2.0 |
| Square | 2.1 | 3.8 |
| Circle | 1.7 | 3.3 |

**Experiment 3: Drawing Quality (house silhouette)**

| Metric | Value |
|---|---|
| Mean directed distance | 11.9 mm |
| 95th-percentile | 21.1 mm |

**Experiment 4: Nullspace Projection Benefit**

| Condition | Joint saturation rate |
|---|---|
| Without nullspace term | 9% |
| With nullspace term | 0% |

## Using Your Own Image (Experiment 3)

1. Place your bitmap (e.g. `my_drawing.png`) in the `code/` folder.
2. Run in a Python session or custom script:

```python
from trajectory_generation import image_to_polylines, polylines_to_3d
from experiments import paper_pose, home_config
from controller import track_waypoints

polylines = image_to_polylines("my_drawing.png", scale_m_per_px=4e-4)
strokes   = polylines_to_3d(polylines, paper_pose())

for stroke in strokes:
    errors, _, _ = track_waypoints(home_config(), stroke['waypoints'])
```

> Requires `opencv-python` to be installed (see above).

## Module Reference

**`kinematics.py`**
- `forward_kinematics(q)` → 4×4 SE(3) transform
- `jacobian(q)` → 6×n Jacobian matrix

**`controller.py`**
- `control_step(q, T_des, ...)` → `(q_new, pos_error)`
- `track_waypoints(q0, wps, ...)` → `(errors, q_final, sat_count)`

**`grasp_planner.py`**
- `grasp_pose(T_pencil)` → 4×4 SE(3) gripper target
- `ik_solve(q0, T_target, ...)` → `(success, q_sol, final_error)`

**`trajectory_generation.py`**
- `image_to_polylines(path, ...)` → list of (M,2) polylines [m]
- `polylines_to_3d(polys, T_paper)` → list of stroke dicts

## Control Parameters (defaults)

| Parameter | Value | Description |
|---|---|---|
| `kv` | 5.0 | Translational gain [1/s] |
| `kw` | 2.0 | Rotational gain [1/s] |
| `dt` | 0.01 | Timestep [s] |
| `lam` | 1e-4 | Pseudoinverse damping |

Tune `kv` / `kw` in `experiments.py` to trade off convergence speed vs. stability near singularities.

## Notes

- The numerical Jacobian uses central finite differences (`eps = 1e-6` rad).
- Joint limits are set to ±120° (±2π/3 rad) to demonstrate nullspace benefit.
- The home configuration `[0, -π/4, π/2, -π/4, -π/2, 0]` is well inside those limits.
- Grasp "success" is defined as IK convergence to within 8 mm of the target position.

