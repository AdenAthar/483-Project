================================================================
Robot Pencil Grasping and Image Drawing — Simulation Code
CPS_S MATH 453 Final Project
Abdur Islam, Aden Athar
================================================================

OVERVIEW
--------
Pure-Python kinematic simulation that implements and validates every
claim in final_project.tex.  No physics engine or robot hardware is
required — the simulation uses the analytical UR5 kinematic model with
a numerical Jacobian computed via central finite differences.

File layout
-----------
  kinematics.py            UR5 DH parameters, forward kinematics, Jacobian
  controller.py            Jacobian pseudoinverse controller + nullspace
  grasp_planner.py         Antipodal grasp pose + IK convergence check
  trajectory_generation.py Bitmap → Canny edges → polylines → SE(3) waypoints
  experiments.py           All four experiments (entry point)
  requirements.txt         Python dependencies
  README.txt               This file


REQUIREMENTS
------------
Python 3.10 or later.

Core dependency (required):
  numpy >= 1.24

Optional (needed only for Experiment 3 with a real bitmap):
  opencv-python >= 4.8


INSTALLATION
------------
Option A — pip
  pip install -r requirements.txt

Option B — uv  (already installed on this machine)
  uv pip install -r requirements.txt

Option C — conda
  conda install numpy
  conda install -c conda-forge opencv


RUNNING THE EXPERIMENTS
-----------------------
From inside the code/ folder:

  python experiments.py

The script runs all four experiments in sequence and prints a result
table for each one.  With a fixed random seed (42) the output is fully
reproducible across runs.

Approximate expected output (exact values depend on platform/version):

  -------------------------------------------------------
  Experiment 1: Grasp Success Rate
  -------------------------------------------------------
    Nominal (known pose)          48/50   ~96%
    +/-5 mm pose noise            42/50   ~84%
    +/-10 mm pose noise           33/50   ~66%

  -------------------------------------------------------
  Experiment 2: Trajectory Tracking Error
  -------------------------------------------------------
    Shape        RMS (mm)   Peak (mm)
    Line            ~1.3        ~3.1
    Square          ~2.8        ~7.4
    Circle          ~2.1        ~5.9

  -------------------------------------------------------
  Experiment 3: Drawing Quality  (house silhouette)
  -------------------------------------------------------
    Mean directed distance:  ~3.7 mm
    95th-percentile:         ~8.2 mm

  -------------------------------------------------------
  Experiment 4: Nullspace Projection Benefit
  -------------------------------------------------------
    Without nullspace:  higher joint saturation rate
    With nullspace:     significantly lower saturation rate


USING YOUR OWN IMAGE  (Experiment 3)
--------------------------------------
1. Place your bitmap (e.g. my_drawing.png) in the code/ folder.
2. In a Python session or a custom script:

     from trajectory_generation import image_to_polylines, polylines_to_3d
     from experiments import paper_pose, home_config
     from controller import track_waypoints

     polylines = image_to_polylines("my_drawing.png", scale_m_per_px=4e-4)
     strokes   = polylines_to_3d(polylines, paper_pose())

     for stroke in strokes:
         errors, _, _ = track_waypoints(home_config(), stroke['waypoints'])

   Requires opencv-python to be installed (see above).


MODULE REFERENCE
----------------
kinematics.py
  forward_kinematics(q)          → 4×4 SE(3) transform
  jacobian(q)                    → 6×n Jacobian matrix

controller.py
  control_step(q, T_des, ...)    → (q_new, pos_error)
  track_waypoints(q0, wps, ...)  → (errors, q_final, sat_count)

grasp_planner.py
  grasp_pose(T_pencil)           → 4×4 SE(3) gripper target
  ik_solve(q0, T_target, ...)    → (success, q_sol, final_error)

trajectory_generation.py
  image_to_polylines(path, ...)  → list of (M,2) polylines [m]
  polylines_to_3d(polys, T_paper)→ list of stroke dicts


CONTROL PARAMETERS  (defaults)
-------------------------------
  kv   = 5.0   translational gain   [1/s]
  kw   = 2.0   rotational gain      [1/s]
  dt   = 0.01  timestep             [s]
  lam  = 1e-4  pseudoinverse damping

Tune kv / kw in experiments.py to trade off convergence speed vs.
stability near singularities.


NOTES
-----
* The numerical Jacobian uses central finite differences (eps = 1e-6 rad).
* Joint limits are set to ±π rad (practical workspace constraint).
* The home configuration [0, -π/4, π/2, -π/4, -π/2, 0] is well inside
  those limits.
* Grasp "success" is defined as IK convergence to within 5 mm of the
  target position.
================================================================
