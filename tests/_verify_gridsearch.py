"""
Grid search for best PD controller parameters per interpolation method.
Mirrors examples/120_PDController_joint_gridsearch.ipynb (cell 5 — full alpha×interp sweep).

For each best-found interpolation config this script produces:
  - controls+error plot
  - metrics plot
  - metrics summary table  (matches LaTeX table layout)
  - PD gains + alpha table  (kp, kd, alpha per interpolation)
Plots saved to tests/gridsearch_comparison/.
"""
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

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
MAX_ERROR_DEG = 0.5

ALPHA_GRID         = [0.5, 0.6, 0.8, 1.0]
INTERPOLATION_GRID = ["linear", "cubic", "trapezoidal", "quintic"]

KP_GRIDS = [
    np.array([5, 20, 60, 150, 400]),
    np.array([5, 20, 60, 150, 400]),
    np.array([5, 20, 60, 150, 400]),
    np.array([1, 5, 15, 40, 100]),
    np.array([1, 5, 15, 40, 100]),
    np.array([1, 5, 15, 40, 100]),
]
RATIO_GRIDS = [np.array([0.05, 0.10, 0.15])] * 6
KP_INIT     = np.array([60.0, 60.0, 60.0, 15.0, 15.0, 15.0])
RATIO_INIT  = np.array([0.10] * 6)

OUT_DIR = Path(__file__).parent / "gridsearch_comparison"
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

# ── evaluate + coordinate descent (mirrors notebook cell 1) ──────────────────
def _evaluate(kp, kd, alpha, interpolation, get_xs_us=False):
    controller = RobotController(robot_pin_model)
    xs, us     = [], []
    try:
        current_q = path[0]
        current_v = np.zeros(nv)
        for i in range(1, len(path)):
            x_start = np.concatenate([current_q, current_v])
            x_goal  = np.concatenate([path[i], np.zeros(nv)])
            t_xs, t_us = controller.compute_control(
                x_start=x_start, x_goal=x_goal,
                T=T, DT=DT, kp=kp, kd=kd,
                alpha=alpha, interpolation=interpolation, option="pd")
            current_q = t_xs[-1][:nq]
            current_v = t_xs[-1][nq:]
            xs.extend(t_xs)
            us.extend(t_us)
    except Exception as e:
        print(f"Solver failed: {e}")
        if get_xs_us:
            return float("inf"), [], []
        return float("inf")

    prec  = tracking_error(xs, path[1:], seg)
    score = unified_cost(us, xs, DT, path[1:], seg)
    if prec > MAX_ERROR_DEG:
        score += 1e9 * (prec - MAX_ERROR_DEG)
    score = float("inf") if np.isnan(score) else float(score)
    if get_xs_us:
        return score, xs, us
    return score


def _coordinate_grid_search(alpha, interpolation):
    kp    = KP_INIT.copy()
    ratio = RATIO_INIT.copy()
    best  = _evaluate(kp, kp * ratio, alpha, interpolation)
    n_eval = 1
    for _ in range(4):
        improved = False
        for j in range(len(kp)):
            for kp_val in KP_GRIDS[j]:
                for r_val in RATIO_GRIDS[j]:
                    t_kp    = kp.copy();    t_kp[j]    = kp_val
                    t_ratio = ratio.copy(); t_ratio[j] = r_val
                    s = _evaluate(t_kp, t_kp * t_ratio, alpha, interpolation)
                    n_eval += 1
                    if s < best:
                        best, kp, ratio, improved = s, t_kp, t_ratio, True
        if not improved:
            break
    return kp, ratio, best, n_eval

# ── grid search across alpha × interpolation ─────────────────────────────────
print("Running coordinate grid search across alpha × interpolation …")
t_search_start = time.time()

best_score      = float("inf")
best_params     = {}
best_per_interp = {interp: {"score": float("inf")} for interp in INTERPOLATION_GRID}

for alpha in ALPHA_GRID:
    for interp in INTERPOLATION_GRID:
        kp_cur, ratio_cur, score_cur, n_eval = _coordinate_grid_search(alpha, interp)
        print(f"  alpha={alpha:.1f}  interp={interp:<12}  score={score_cur:.4e}  evals={n_eval}")
        entry = dict(alpha=alpha, interpolation=interp,
                     score=score_cur, kp=kp_cur.copy(), ratio=ratio_cur.copy())
        if score_cur < best_per_interp[interp]["score"]:
            best_per_interp[interp] = entry
        if score_cur < best_score:
            best_score  = score_cur
            best_params = entry

search_time = time.time() - t_search_start
print(f"\nGrid search done in {search_time:.1f}s")
print(f"Global best: alpha={best_params['alpha']}  interp={best_params['interpolation']}  score={best_score:.4e}")

# ── evaluate best config per interpolation ────────────────────────────────────
summary      = []
gains_summary = []

for interp in INTERPOLATION_GRID:
    r = best_per_interp[interp]
    if r["score"] == float("inf"):
        continue

    kp_best = r["kp"]
    kd_best = kp_best * r["ratio"]

    print(f"\n{'=' * 60}")
    print(f"  {interp}  (alpha={r['alpha']})")
    print(f"{'=' * 60}")

    _, xs, us = _evaluate(kp_best, kd_best, r["alpha"], interp, get_xs_us=True)
    robot_mj_model.reset()
    xs_real, us_real = robot_mj_model.simulate_control(
        us=us, xs=xs, kp=kp_best, kd=kd_best)

    slug = f"gs_{interp}"

    with _save_plots(f"{slug}_controls_error"):
        plot_controls_and_error(
            us_real, xs_real, DT, path[1:], seg,
            title=f"Grid Search — {interp}  (α={r['alpha']})")

    with _save_plots(f"{slug}_metrics"):
        plot_metrics(us_real, xs_real, DT, path[1:], seg)

    us_arr = np.asarray(us_real)
    xs_arr = np.asarray(xs_real)
    us_ff  = np.asarray(us, dtype=float)
    u_fb   = us_arr - us_ff
    g_tot  = np.linalg.norm(us_arr)
    ff_pct = np.linalg.norm(us_ff) / (g_tot or 1.0) * 100.0
    fb_pct = np.linalg.norm(u_fb)  / (g_tot or 1.0) * 100.0

    summary.append(dict(
        interpolation = interp,
        E_thermal     = thermal_energy(us_arr, DT),
        E_mech        = mechanical_work(us_arr, xs_arr, DT),
        P_peak        = peak_power(us_arr, xs_arr),
        tracking_err  = tracking_error(xs_arr, path[1:], seg),
        torque_jerk   = torque_jerk(us_arr),
        ff_pct        = ff_pct,
        fb_pct        = fb_pct,
        unified_cost  = unified_cost(us_arr, xs_arr, DT, path[1:], seg),
        ctrl_time_s   = search_time,
    ))

    gains_summary.append(dict(
        interpolation = interp,
        alpha         = r["alpha"],
        kp            = kp_best,
        kd            = kd_best,
        score         = unified_cost(us_arr, xs_arr, DT, path[1:], seg),
    ))

# ── Table 1: metrics ─────────────────────────────────────────────────────────
W = 125
print(f"\n\n{'=' * W}")
print(f"  METRICS SUMMARY   (coordinate grid search, n_passes=4)")
print(f"{'=' * W}")
print(f"{'Method':<14} {'E_therm[N²m²s]':>16} {'E_mech[J]':>11} "
      f"{'P_peak[W]':>11} {'Err[deg]':>10} {'Jerk[N²m²/s]':>14} "
      f"{'FF[%]':>7} {'FB[%]':>7} {'UnifCost':>14}")
print("-" * W)
for r in summary:
    print(
        f"{r['interpolation']:<14} "
        f"{r['E_thermal']:>16.2f} "
        f"{r['E_mech']:>11.2f} "
        f"{r['P_peak']:>11.2f} "
        f"{r['tracking_err']:>10.4f} "
        f"{r['torque_jerk']:>14.2f} "
        f"{r['ff_pct']:>7.1f} "
        f"{r['fb_pct']:>7.1f} "
        f"{r['unified_cost']:>14.4e}"
    )
print("=" * W)

# ── Table 2: PD gains + alpha ─────────────────────────────────────────────────
print(f"\n\n{'=' * W}")
print(f"  BEST PD GAINS PER INTERPOLATION METHOD")
print(f"{'=' * W}")
print(f"{'Method':<14} {'alpha':>6} {'score':>14}  {'kp (J1..J6)':<42}  {'kd (J1..J6)'}")
print("-" * W)
for g in gains_summary:
    kp_str = "[" + ", ".join(f"{v:.1f}" for v in g["kp"]) + "]"
    kd_str = "[" + ", ".join(f"{v:.3f}" for v in g["kd"]) + "]"
    print(
        f"{g['interpolation']:<14} "
        f"{g['alpha']:>6.1f} "
        f"{g['score']:>14.4e}  "
        f"{kp_str:<42}  "
        f"{kd_str}"
    )
print("=" * W)
print(f"\nPlots saved to: {OUT_DIR.resolve()}")
