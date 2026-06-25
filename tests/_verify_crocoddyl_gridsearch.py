"""
Grid search for best Crocoddyl advanced cost weights.
Mirrors examples/200_Crocoddyl_joint.ipynb.

Performs coordinate descent over cost weights (track_weight, ctrl_weight,
terminal_pose_weight, v_weight, acc_weight), evaluating unified_cost on the
closed-loop MuJoCo trajectory.

For the best-found config this script produces:
  - controls+error plot
  - metrics plot
  - metrics summary table
  - cost weight table
Plots saved to tests/crocoddyl_gridsearch/.
"""
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import crocoddyl

from optimal.RobotController import RobotController
from optimal.RobotMujocoModel import RobotMujocoModel
from optimal.RobotPinocchioModel import RobotPinocchioModel
from optimal.configuration import UR10_sandbox, Trajectories
from optimal.metrics import (
    thermal_energy, mechanical_work, peak_power,
    tracking_error, torque_jerk, plot_metrics,
    plot_controls_and_error, unified_cost,
)

# ── setup ────────────────────────────────────────────────────────────────────
cfg             = UR10_sandbox
robot_mj_model  = RobotMujocoModel(cfg.mjcf_path, cfg.list_of_joints)
robot_pin_model = RobotPinocchioModel(cfg.urdf_path, cfg.list_of_joints)

DT            = robot_mj_model.model.opt.timestep
path          = Trajectories.joint_6dof
T             = 1000
seg           = [T] * (len(path) - 1)
nq            = len(robot_pin_model.q_indices)
nv            = len(robot_pin_model.v_indices)
MAX_ERROR_DEG = 2.0   # Crocoddyl trades terminal precision for smoothness; relax vs PD (0.5 deg)

# PD warm-start gains (fixed — same as notebook 200_)
# KP = [1414.94,1693.23,1202.19,7.92,34.51,1.24]
# KD = [204.882,113.588,148.053,0.687,3.741,0.178]
KP = cfg.kp
KD = cfg.kd
ALPHA         = 0.9
INTERPOLATION = "quintic"

# Starting weights — best found without track_interpolated, used as warm start.
INIT_WEIGHTS = dict(
    track_weight          = 1e4,
    ctrl_weight           = 1e3,
    terminal_pose_weight  = 1e7,
    v_weight              = 1e-1,
    acc_weight            = 1e1,
)

# Absolute grids centered on the known best values
TRACK_WEIGHT_GRID    = np.array([3e3, 5e3, 1e4, 3e4, 1e5, 3e5])
CTRL_WEIGHT_GRID     = np.array([1e2, 3e2, 1e3, 3e3, 1e4])
TERMINAL_WEIGHT_GRID = np.array([3e6, 1e7, 3e7, 1e8, 3e8])
V_WEIGHT_GRID        = np.array([1e-2, 5e-2, 1e-1, 3e-1, 1e0])
ACC_WEIGHT_GRID      = np.array([1e0, 5e0, 1e1, 3e1, 1e2])

OUT_DIR = Path(__file__).parent / "crocoddyl_gridsearch"
OUT_DIR.mkdir(exist_ok=True)

# ── plot saving ───────────────────────────────────────────────────────────────
@contextmanager
def _save_plots(prefix: str):
    counter   = [0]
    orig_show = plt.show

    def _show_and_save():
        for fnum in plt.get_fignums():
            counter[0] += 1
            plt.figure(fnum).savefig(
                OUT_DIR / f"{prefix}_{counter[0]:02d}.png",
                bbox_inches="tight", dpi=150,
            )
        orig_show()

    plt.show = _show_and_save
    try:
        yield
    finally:
        plt.show = orig_show

# ── evaluate ─────────────────────────────────────────────────────────────────
def _evaluate(weights: dict, get_xs_us: bool = False):
    controller = RobotController(robot_pin_model)
    xs, us     = [], []
    try:
        current_q = path[0].copy()
        current_v = np.zeros(nv)
        for i in range(1, len(path)):
            x_start = np.concatenate([current_q, current_v])
            x_goal  = np.concatenate([path[i], np.zeros(nv)])
            t_xs, t_us = controller.compute_control(
                x_start=x_start, x_goal=x_goal,
                T=T, DT=DT,
                kp=KP, kd=KD,
                alpha=ALPHA, interpolation=INTERPOLATION,
                option="crocoddyl_advanced",
                solver=crocoddyl.SolverBoxFDDP,
                # track_interpolated=True,
                **weights,
            )
            current_q = t_xs[-1][:nq]
            current_v = t_xs[-1][nq:]
            xs.extend(t_xs)
            us.extend(t_us)
    except Exception as e:
        print(f"  Solver failed: {e}")
        if get_xs_us:
            return float("inf"), [], []
        return float("inf")

    # Score on Pinocchio-integrated trajectory (fast, no MuJoCo needed during search)
    prec  = tracking_error(xs, path[1:], seg)
    score = unified_cost(np.asarray(us), np.asarray(xs), DT, path[1:], seg)
    if prec > MAX_ERROR_DEG:
        score += 1e9 * (prec - MAX_ERROR_DEG)
    score = float("inf") if np.isnan(score) else float(score)

    if get_xs_us:
        return score, xs, us
    return score


# ── coordinate grid search over cost weights ─────────────────────────────────
PARAM_GRIDS = [
    ("track_weight",         TRACK_WEIGHT_GRID),
    ("ctrl_weight",          CTRL_WEIGHT_GRID),
    ("terminal_pose_weight", TERMINAL_WEIGHT_GRID),
    ("v_weight",             V_WEIGHT_GRID),
    ("acc_weight",           ACC_WEIGHT_GRID),
]

def _coordinate_grid_search(n_passes=3):
    weights = INIT_WEIGHTS.copy()
    best    = _evaluate(weights)
    n_eval  = 1
    print(f"  Initial score: {best:.4e}  weights={weights}")

    for p in range(n_passes):
        improved = False
        for param, grid in PARAM_GRIDS:
            for val in grid:
                candidate = {**weights, param: float(val)}
                s = _evaluate(candidate)
                n_eval += 1
                if s < best:
                    best      = s
                    weights   = candidate.copy()
                    improved  = True
                    print(f"    [{param}={val:.1e}]  score={s:.4e}")
        print(f"  Pass {p + 1}: best={best:.4e}  (evals={n_eval})")
        if not improved:
            break

    return weights, best, n_eval


# ── run ───────────────────────────────────────────────────────────────────────
print("Running coordinate grid search over Crocoddyl cost weights …")
t_start = time.time()

best_weights, best_score, n_eval = _coordinate_grid_search(n_passes=3)

search_time = time.time() - t_start
print(f"\nDone in {search_time:.1f}s   total evals={n_eval}")
print(f"Best score (Pinocchio open-loop): {best_score:.4e}")
print("Best weights:")
for k, v in best_weights.items():
    print(f"  {k:<24} = {v:.2e}")

# ── final evaluation with plots (MuJoCo closed-loop, run once) ───────────────
print("\nFinal evaluation …")
_, xs, us = _evaluate(best_weights, get_xs_us=True)
robot_mj_model.reset()
xs_real, us_real = robot_mj_model.simulate_control(us=us, xs=xs, kp=KP, kd=KD)

us_arr = np.asarray(us_real)
xs_arr = np.asarray(xs_real)
us_ff  = np.asarray(us, dtype=float)
u_fb   = us_arr - us_ff
g_tot  = np.linalg.norm(us_arr)
ff_pct = np.linalg.norm(us_ff) / (g_tot or 1.0) * 100.0
fb_pct = np.linalg.norm(u_fb)  / (g_tot or 1.0) * 100.0

with _save_plots("croc_controls_error"):
    plot_controls_and_error(
        us_arr, xs_arr, DT, path[1:], seg,
        title=f"Crocoddyl Advanced — Grid Search Best")

with _save_plots("croc_metrics"):
    plot_metrics(us_arr, xs_arr, DT, path[1:], seg, controller_time=search_time)

# ── Table 1: metrics ─────────────────────────────────────────────────────────
W = 115
cost = unified_cost(us_arr, xs_arr, DT, path[1:], seg)

print(f"\n\n{'=' * W}")
print(f"  METRICS SUMMARY   (Crocoddyl advanced, coordinate grid search, n_passes=3)")
print(f"{'=' * W}")
print(f"{'E_therm[N²m²s]':>16} {'E_mech[J]':>11} {'P_peak[W]':>11} "
      f"{'Err[deg]':>10} {'Jerk[N²m²/s]':>14} {'FF[%]':>7} {'FB[%]':>7} {'UnifCost':>14}")
print("-" * W)
print(f"{thermal_energy(us_arr, DT):>16.4e} "
      f"{mechanical_work(us_arr, xs_arr, DT):>11.4e} "
      f"{peak_power(us_arr, xs_arr):>11.4e} "
      f"{tracking_error(xs_arr, path[1:], seg):>10.4f} "
      f"{torque_jerk(us_arr):>14.4e} "
      f"{ff_pct:>7.1f} "
      f"{fb_pct:>7.1f} "
      f"{cost:>14.4e}")
print("=" * W)

# ── Table 2: best cost weights ────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"  BEST COST WEIGHTS")
print(f"{'=' * 70}")
print(f"{'Parameter':<26} {'Value':>12}   (search grid)")
print("-" * 70)
grid_map = dict(PARAM_GRIDS)
for k, v in best_weights.items():
    grid_str = "  [" + ", ".join(f"{g:.0e}" for g in grid_map[k]) + "]" if k in grid_map else ""
    print(f"  {k:<24} {v:>12.2e}{grid_str}")
print(f"{'=' * 70}")
print(f"  Search time: {search_time:.1f}s   evals: {n_eval}   score: {best_score:.4e}")
print(f"{'=' * 70}")
