"""
Optuna hyperparameter search for Crocoddyl advanced cost weights.
Mirrors examples/200_Crocoddyl_joint.ipynb.

Axes of comparison:
  - Sampler:              TPE  |  CMA-ES
  - Interpolation:        with (quintic, alpha=0.9)  |  without (alpha=0.0)
  - Solver:               DDP  |  FDDP  |  BoxFDDP

Each combination is treated as an independent strategy.
Score is always unified_cost on the closed-loop MuJoCo trajectory.

Output:
  - controls+error plot and metrics plot per strategy
  - summary table: all strategies ranked
  - best-per-category breakdown
Plots saved to tests/crocoddyl_optuna/.
"""
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import CmaEsSampler
import crocoddyl

optuna.logging.set_verbosity(optuna.logging.WARNING)

from optimal.RobotController import RobotController
from optimal.RobotMujocoModel import RobotMujocoModel
from optimal.RobotPinocchioModel import RobotPinocchioModel
from optimal.configuration import UR10_sandbox, Trajectories, UR10_weighted
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
path          = Trajectories.joint_weighted
T             = 1000
seg           = [T] * (len(path) - 1)
nq            = len(robot_pin_model.q_indices)
nv            = len(robot_pin_model.v_indices)

MAX_ERROR_DEG = 2.0   # relaxed for Crocoddyl — it trades terminal precision for smoothness
N_TRIALS      = 50    # per strategy; increase for better convergence
SEED          = 42

# PD warm-start gains — best found by _verify_crocoddyl_gridsearch.py
# KP = [1414.94,1693.23,1202.19,7.92,34.51,1.24]
# KD = [204.882,113.588,148.053,0.687,3.741,0.178]
KP = cfg.kp
KD = cfg.kd
WARM_ALPHA        = 0.9
WARM_INTERPOLATION = "quintic"

# Reference weights (used as log-scale center for Optuna search)
REF_WEIGHTS = dict(
    track_weight          = 1e4,
    ctrl_weight           = 1e3,
    terminal_pose_weight  = 1e7,
    v_weight              = 1e-1,
    acc_weight            = 1e1,
    terminal_v_weight     = 1e-1,
)

OUT_DIR = Path(__file__).parent / "crocoddyl_optuna"
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
def _evaluate(weights: dict, alpha: float, interpolation: str,
              solver_cls, get_xs_us: bool = False):
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
                alpha=alpha, interpolation=interpolation,
                option="crocoddyl_advanced",
                solver=solver_cls,
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

# ── Optuna objective ─────────────────────────────────────────────────────────
def _make_objective(track_interpolated: bool, solver_cls):
    def objective(trial):
        # Search as ratios relative to REF_WEIGHTS — keeps all axes on the same [0.1, 10] scale
        track_w      = trial.suggest_float("track_ratio",      0.1, 10.0, log=True) * REF_WEIGHTS["track_weight"]
        ctrl_w       = trial.suggest_float("ctrl_ratio",       0.1, 10.0, log=True) * REF_WEIGHTS["ctrl_weight"]
        terminal_w   = trial.suggest_float("terminal_ratio",   0.1, 10.0, log=True) * REF_WEIGHTS["terminal_pose_weight"]
        v_w          = trial.suggest_float("v_ratio",          0.1, 10.0, log=True) * REF_WEIGHTS["v_weight"]
        acc_w        = trial.suggest_float("acc_ratio",        0.1, 10.0, log=True) * REF_WEIGHTS["acc_weight"]
        terminal_v_w = trial.suggest_float("terminal_v_ratio", 0.1, 10.0, log=True) * REF_WEIGHTS["terminal_v_weight"]

        weights = dict(
            track_weight         = track_w,
            ctrl_weight          = ctrl_w,
            terminal_pose_weight = terminal_w,
            v_weight             = v_w,
            acc_weight           = acc_w,
            terminal_v_weight    = terminal_v_w,
            track_interpolated   = track_interpolated,
        )
        return _evaluate(weights, WARM_ALPHA, WARM_INTERPOLATION, solver_cls)
    return objective

def _params_from_trial(best_params: dict) -> dict:
    """Convert Optuna best_params (ratios) back to actual weight values."""
    return dict(
        track_weight         = best_params["track_ratio"]      * REF_WEIGHTS["track_weight"],
        ctrl_weight          = best_params["ctrl_ratio"]       * REF_WEIGHTS["ctrl_weight"],
        terminal_pose_weight = best_params["terminal_ratio"]   * REF_WEIGHTS["terminal_pose_weight"],
        v_weight             = best_params["v_ratio"]          * REF_WEIGHTS["v_weight"],
        acc_weight           = best_params["acc_ratio"]        * REF_WEIGHTS["acc_weight"],
        terminal_v_weight    = best_params["terminal_v_ratio"] * REF_WEIGHTS["terminal_v_weight"],
    )

# ── strategy runner ──────────────────────────────────────────────────────────
def _run_strategy(sampler, track_interpolated: bool, solver_cls):
    t0    = time.time()
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        _make_objective(track_interpolated, solver_cls),
        n_trials=N_TRIALS, show_progress_bar=True,
    )
    opt_time = time.time() - t0

    best_weights = _params_from_trial(study.best_params)
    best_weights["track_interpolated"] = track_interpolated
    score, xs, us = _evaluate(
        best_weights, WARM_ALPHA, WARM_INTERPOLATION, solver_cls, get_xs_us=True)
    # Final MuJoCo closed-loop run for plots and summary metrics
    robot_mj_model.reset()
    xs_real, us_real = robot_mj_model.simulate_control(us=us, xs=xs, kp=KP, kd=KD)
    return xs, us, xs_real, us_real, opt_time, score, best_weights

# ── strategy definitions ─────────────────────────────────────────────────────
SOLVER_MAP = {
    # "DDP":     crocoddyl.SolverDDP,
    # "FDDP":    crocoddyl.SolverFDDP,
    "BoxFDDP": crocoddyl.SolverBoxFDDP,
}

STRATEGIES = []
for sampler_name, sampler_fn in [
    ("TPE",    lambda: optuna.samplers.TPESampler(seed=SEED)),
    ("CMA-ES", lambda: CmaEsSampler(seed=SEED, warn_independent_sampling=False)),
]:
    for interp_label, track_interpolated in [
        ("track-interp",    True),   # per-timestep Crocoddyl reference from PD warm-start
        # ("no-track-interp", False),  # fixed endpoint reference (standard)
    ]:
        for solver_name, solver_cls in SOLVER_MAP.items():
            label = f"{sampler_name} | {interp_label} | {solver_name}"
            STRATEGIES.append((label, sampler_name, interp_label, solver_name,
                                track_interpolated, solver_cls, sampler_fn))

# ── main loop ─────────────────────────────────────────────────────────────────
summary = []

for label, sampler_name, interp_label, solver_name, track_interpolated, solver_cls, sampler_fn in STRATEGIES:
    print(f"\n{'=' * 70}")
    print(f"  {label}   (n_trials={N_TRIALS})")
    print(f"{'=' * 70}")

    xs, us, xs_real, us_real, opt_time, score, best_weights = _run_strategy(
        sampler_fn(), track_interpolated, solver_cls)

    us_arr = np.asarray(us_real)
    xs_arr = np.asarray(xs_real)
    us_ff  = np.asarray(us, dtype=float)
    u_fb   = us_arr - us_ff
    g_tot  = np.linalg.norm(us_arr)
    ff_pct = np.linalg.norm(us_ff) / (g_tot or 1.0) * 100.0
    fb_pct = np.linalg.norm(u_fb)  / (g_tot or 1.0) * 100.0
    cost   = unified_cost(us_arr, xs_arr, DT, path[1:], seg)

    slug = label.replace(" ", "_").replace("|", "").replace("__", "_").strip("_")

    with _save_plots(f"{slug}_controls_error"):
        plot_controls_and_error(us_arr, xs_arr, DT, path[1:], seg,
                                title=f"Crocoddyl — {label}")

    with _save_plots(f"{slug}_metrics"):
        plot_metrics(us_arr, xs_arr, DT, path[1:], seg, controller_time=opt_time)

    summary.append(dict(
        label        = label,
        sampler      = sampler_name,
        interpolated = interp_label,
        solver       = solver_name,
        E_thermal    = thermal_energy(us_arr, DT),
        E_mech       = mechanical_work(us_arr, xs_arr, DT),
        P_peak       = peak_power(us_arr, xs_arr),
        tracking_err = tracking_error(xs_arr, path[1:], seg),
        torque_jerk  = torque_jerk(us_arr),
        ff_pct       = ff_pct,
        fb_pct       = fb_pct,
        unified_cost = cost,
        opt_time_s   = opt_time,
        best_weights = best_weights,
    ))

# ── Table 1: full summary ─────────────────────────────────────────────────────
W = 155
summary_sorted = sorted(summary, key=lambda r: r["unified_cost"])

print(f"\n\n{'=' * W}")
print(f"  FULL SUMMARY   (n_trials={N_TRIALS} per strategy, sorted by UnifCost)")
print(f"{'=' * W}")
print(f"{'Strategy':<38} {'E_therm':>14} {'E_mech[J]':>11} {'P_peak[W]':>11} "
      f"{'Err[deg]':>10} {'Jerk':>12} {'FF%':>6} {'FB%':>6} {'UnifCost':>14} {'t[s]':>8}")
print("-" * W)
for r in summary_sorted:
    print(f"{r['label']:<38} "
          f"{r['E_thermal']:>14.4e} "
          f"{r['E_mech']:>11.4e} "
          f"{r['P_peak']:>11.4e} "
          f"{r['tracking_err']:>10.4f} "
          f"{r['torque_jerk']:>12.4e} "
          f"{r['ff_pct']:>6.1f} "
          f"{r['fb_pct']:>6.1f} "
          f"{r['unified_cost']:>14.4e} "
          f"{r['opt_time_s']:>8.1f}")
print("=" * W)

# ── Table 2: best per category ────────────────────────────────────────────────
def _best(rows, key, val):
    filtered = [r for r in rows if r[key] == val]
    return min(filtered, key=lambda r: r["unified_cost"]) if filtered else None

categories = [
    ("Sampler",       "sampler",      ["TPE", "CMA-ES"]),
    ("Interpolation", "interpolated", ["track-interp", "no-track-interp"]),
    ("Solver",        "solver",       ["BoxFDDP"]),
]

print(f"\n\n{'=' * W}")
print(f"  BEST PER CATEGORY")
print(f"{'=' * W}")
for cat_name, key, values in categories:
    print(f"\n  {cat_name}:")
    print(f"  {'Value':<12} {'Best strategy':<38} {'UnifCost':>14} {'Err[deg]':>10}")
    print("  " + "-" * 76)
    for val in values:
        r = _best(summary, key, val)
        if r:
            print(f"  {val:<12} {r['label']:<38} {r['unified_cost']:>14.4e} {r['tracking_err']:>10.4f}")

print(f"\n  Overall best:")
best_overall = summary_sorted[0]
print(f"    {best_overall['label']}")
print(f"    UnifCost={best_overall['unified_cost']:.4e}   Err={best_overall['tracking_err']:.4f} deg")
print(f"    Weights: " + "  ".join(f"{k}={v:.2e}" for k, v in best_overall["best_weights"].items()))
print(f"{'=' * W}")

# ── Table 3: cost weights per strategy ───────────────────────────────────────
print(f"\n\n{'=' * W}")
print(f"  BEST COST WEIGHTS PER STRATEGY")
print(f"{'=' * W}")
print(f"{'Strategy':<38} {'track_w':>10} {'ctrl_w':>10} {'term_w':>10} {'v_w':>8} {'acc_w':>8} {'term_v_w':>10}")
print("-" * W)
for r in summary_sorted:
    w = r["best_weights"]
    print(f"{r['label']:<38} "
          f"{w['track_weight']:>10.2e} "
          f"{w['ctrl_weight']:>10.2e} "
          f"{w['terminal_pose_weight']:>10.2e} "
          f"{w['v_weight']:>8.2e} "
          f"{w['acc_weight']:>8.2e} "
          f"{w['terminal_v_weight']:>10.2e}")
print("=" * W)
print(f"\nPlots saved to: {OUT_DIR.resolve()}")
