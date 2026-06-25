"""
Compare Optuna PD controller tuning strategies:
  1. TPE (default) — one global parameter set for the entire path
  2. CMA-ES        — one global parameter set for the entire path
  3. TPE           — individual parameters tuned per segment
  4. CMA-ES        — individual parameters tuned per segment

Mirrors the setup from examples/130_PDController_joint_optuna.ipynb.
Produces per-strategy plots (controls+error, metrics) saved to tests/optuna_strategies/
and prints a final summary table.
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

optuna.logging.set_verbosity(optuna.logging.WARNING)

from optimal.RobotController import RobotController
from optimal.RobotMujocoModel import RobotMujocoModel
from optimal.RobotPinocchioModel import RobotPinocchioModel
from optimal.configuration import UR10_weighted, Trajectories
from optimal.metrics import (
    thermal_energy, mechanical_work, peak_power,
    tracking_error, torque_jerk, plot_metrics,
    plot_controls_and_error, unified_cost,
)

# ── setup ────────────────────────────────────────────────────────────────────
cfg             = UR10_weighted
robot_mj_model  = RobotMujocoModel(cfg.mjcf_path, cfg.list_of_joints)
robot_pin_model = RobotPinocchioModel(cfg.urdf_path, cfg.list_of_joints)
payload         = robot_mj_model.get_obj_payload("box_2")
robot_pin_model.add_payload("robotiq_85_base_link", payload)

DT           = robot_mj_model.model.opt.timestep
path         = Trajectories.joint_6dof
T            = 1000
seg          = [T] * (len(path) - 1)
nq           = len(robot_pin_model.q_indices)
nv           = len(robot_pin_model.v_indices)
MAX_ERROR_DEG = 0.5
N_TRIALS      = 200   # increase for better results; 200 is a reasonable starting point
SEED          = 42    # fixed seed for reproducibility

OUT_DIR = Path(__file__).parent / "optuna_strategies"
OUT_DIR.mkdir(exist_ok=True)

# ── plot saving ──────────────────────────────────────────────────────────────
@contextmanager
def _save_plots(prefix: str):
    counter  = [0]
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

# ── Optuna objective (single path or sub-path) ───────────────────────────────
def _objective(trial, sub_path):
    kp1 = trial.suggest_float("kp1", 1,    2e3, log=True)
    kp2 = trial.suggest_float("kp2", 1,    2e3, log=True)
    kp3 = trial.suggest_float("kp3", 1,    2e3, log=True)
    kp4 = trial.suggest_float("kp4", 1,    50,  log=True)
    kp5 = trial.suggest_float("kp5", 1,    50,  log=True)
    kp6 = trial.suggest_float("kp6", 1,    50,  log=True)
    ratio1 = trial.suggest_float("ratio1", 0.05, 0.15)
    ratio2 = trial.suggest_float("ratio2", 0.05, 0.15)
    ratio3 = trial.suggest_float("ratio3", 0.05, 0.15)
    ratio4 = trial.suggest_float("ratio4", 0.05, 0.15)
    ratio5 = trial.suggest_float("ratio5", 0.05, 0.15)
    ratio6 = trial.suggest_float("ratio6", 0.05, 0.15)
    alpha  = trial.suggest_float("alpha",  0.4,  1.0)
    interp = trial.suggest_categorical("interpolation",
                                       ["linear", "trapezoidal", "cubic", "quintic"])

    kp = np.array([kp1, kp2, kp3, kp4, kp5, kp6])
    kd = np.array([kp1*ratio1, kp2*ratio2, kp3*ratio3,
                   kp4*ratio4, kp5*ratio5, kp6*ratio6])

    controller = RobotController(robot_pin_model)
    xs, us = [], []
    try:
        current_q = sub_path[0]
        current_v = np.zeros(nv)
        for i in range(1, len(sub_path)):
            x_start = np.concatenate([current_q, current_v])
            x_goal  = np.concatenate([sub_path[i], np.zeros(nv)])
            t_xs, t_us = controller.compute_control(
                x_start=x_start, x_goal=x_goal,
                T=T, DT=DT, kp=kp, kd=kd,
                alpha=alpha, interpolation=interp, option="pd")
            current_q = t_xs[-1][:nq]
            current_v = t_xs[-1][nq:]
            xs.extend(t_xs)
            us.extend(t_us)
    except Exception:
        return float("inf")

    seg_lens = [T] * (len(sub_path) - 1)
    score = unified_cost(us, xs, DT, sub_path[1:], seg_lens)
    prec  = tracking_error(xs, sub_path[1:], seg_lens)
    if prec > MAX_ERROR_DEG:
        score += 1e9 * (prec - MAX_ERROR_DEG)
    return float("inf") if np.isnan(score) else float(score)


def _run_trajectory(params_list):
    """Execute the path using a list of per-segment param dicts. Returns xs, us."""
    controller = RobotController(robot_pin_model)
    xs, us    = [], []
    current_q = path[0]
    current_v = np.zeros(nv)
    for i, p in enumerate(params_list):
        kp = np.array([p["kp1"], p["kp2"], p["kp3"], p["kp4"], p["kp5"], p["kp6"]])
        kd = np.array([p["kp1"]*p["ratio1"], p["kp2"]*p["ratio2"], p["kp3"]*p["ratio3"],
                       p["kp4"]*p["ratio4"], p["kp5"]*p["ratio5"], p["kp6"]*p["ratio6"]])
        x_start = np.concatenate([current_q, current_v])
        x_goal  = np.concatenate([path[i + 1], np.zeros(nv)])
        t_xs, t_us = controller.compute_control(
            x_start=x_start, x_goal=x_goal,
            T=T, DT=DT, kp=kp, kd=kd,
            alpha=p["alpha"], interpolation=p["interpolation"], option="pd")
        current_q = t_xs[-1][:nq]
        current_v = t_xs[-1][nq:]
        xs.extend(t_xs)
        us.extend(t_us)
    return xs, us


# ── strategies ───────────────────────────────────────────────────────────────
def _global_strategy(sampler, label):
    """Tune one parameter set for the entire path."""
    t0    = time.time()
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(lambda trial: _objective(trial, path),
                   n_trials=N_TRIALS, show_progress_bar=True)
    opt_time = time.time() - t0

    best      = study.best_params
    params    = [best] * (len(path) - 1)
    xs, us    = _run_trajectory(params)
    robot_mj_model.reset()
    xs_real, us_real = robot_mj_model.simulate_control(
        us=us, xs=xs,
        # kp=np.array([best["kp1"], best["kp2"], best["kp3"],
        #              best["kp4"], best["kp5"], best["kp6"]]),
        # kd=np.array([best["kp1"]*best["ratio1"], best["kp2"]*best["ratio2"],
        #              best["kp3"]*best["ratio3"], best["kp4"]*best["ratio4"],
        #              best["kp5"]*best["ratio5"], best["kp6"]*best["ratio6"]]),
        kp=cfg.kp, kd=cfg.kd
    )
    return xs, us, xs_real, us_real, opt_time, study.best_value, best


def _per_segment_strategy(sampler, label):
    """Tune a separate parameter set for each individual segment."""
    t0          = time.time()
    params_list = []
    current_q   = path[0]
    for i in range(1, len(path)):
        sub_path = [current_q, path[i]]
        study    = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(lambda trial, sp=sub_path: _objective(trial, sp),
                       n_trials=N_TRIALS // (len(path) - 1),
                       show_progress_bar=True)
        params_list.append(study.best_params)
        current_q = path[i]
    opt_time = time.time() - t0

    xs, us = _run_trajectory(params_list)
    # Use gains from first segment for simulate_control (they drive the closed-loop gains)
    p0 = params_list[0]
    robot_mj_model.reset()
    xs_real, us_real = robot_mj_model.simulate_control(
        us=us, xs=xs,
        # kp=np.array([p0["kp1"], p0["kp2"], p0["kp3"],
        #              p0["kp4"], p0["kp5"], p0["kp6"]]),
        # kd=np.array([p0["kp1"]*p0["ratio1"], p0["kp2"]*p0["ratio2"],
        #              p0["kp3"]*p0["ratio3"], p0["kp4"]*p0["ratio4"],
        #              p0["kp5"]*p0["ratio5"], p0["kp6"]*p0["ratio6"]]),
        kp=cfg.kp, kd=cfg.kd
    )
    score = unified_cost(np.asarray(us_real), np.asarray(xs_real), DT, path[1:], seg)
    # For per-segment, report the first segment's params as representative global gains
    return xs, us, xs_real, us_real, opt_time, score, params_list


# ── main ─────────────────────────────────────────────────────────────────────
STRATEGIES = [
    ("TPE — global",         lambda: _global_strategy(      optuna.samplers.TPESampler(seed=SEED),                      "TPE global")),
    ("CMA-ES — global",      lambda: _global_strategy(      CmaEsSampler(seed=SEED, warn_independent_sampling=False),   "CMA-ES global")),
    ("TPE — per segment",    lambda: _per_segment_strategy( optuna.samplers.TPESampler(seed=SEED),                      "TPE per-seg")),
    ("CMA-ES — per segment", lambda: _per_segment_strategy( CmaEsSampler(seed=SEED, warn_independent_sampling=False),   "CMA-ES per-seg")),
]

summary = []

for label, run_fn in STRATEGIES:
    print(f"\n{'=' * 65}")
    print(f"  Strategy: {label}   (n_trials={N_TRIALS})")
    print(f"{'=' * 65}")

    xs, us, xs_real, us_real, opt_time, best_score, best_params = run_fn()

    us_arr = np.asarray(us_real)
    xs_arr = np.asarray(xs_real)

    slug = label.replace(" ", "_").replace("—", "").replace("__", "_").strip("_")

    print(f"\n[{label}] controls + error:")
    with _save_plots(f"{slug}_controls_error"):
        plot_controls_and_error(us_arr, xs_arr, DT, path[1:], seg,
                                title=f"PD Control — {label}")

    print(f"\n[{label}] metrics:")
    with _save_plots(f"{slug}_metrics"):
        plot_metrics(us_arr, xs_arr, DT, path[1:], seg, controller_time=opt_time)

    E_th   = thermal_energy(us_arr, DT)
    E_mech = mechanical_work(us_arr, xs_arr, DT)
    P_pk   = peak_power(us_arr, xs_arr)
    t_err  = tracking_error(xs_arr, path[1:], seg)
    t_jerk = torque_jerk(us_arr)
    cost   = unified_cost(us_arr, xs_arr, DT, path[1:], seg)

    # FF / FB percentages (same logic as plot_feedback_part)
    us_ff  = np.asarray(us, dtype=float)
    u_fb   = us_arr - us_ff[:len(us_arr)]
    g_tot  = np.linalg.norm(us_arr)
    g_ff   = np.linalg.norm(us_ff[:len(us_arr)])
    g_fb   = np.linalg.norm(u_fb)
    denom  = g_tot if g_tot > 0 else 1.0
    ff_pct = g_ff / denom * 100.0
    fb_pct = g_fb / denom * 100.0

    cost   = unified_cost(us_arr, xs_arr, DT, path[1:], seg)

    # Represent params: for global → single dict, for per-segment → list of dicts
    params_repr = best_params if isinstance(best_params, list) else [best_params]

    summary.append(dict(
        label        = label,
        E_thermal    = E_th,
        E_mech       = E_mech,
        P_peak       = P_pk,
        tracking_err = t_err,
        torque_jerk  = t_jerk,
        ff_pct       = ff_pct,
        fb_pct       = fb_pct,
        unified_cost = cost,
        opt_time_s   = opt_time,
        params_repr  = params_repr,
    ))

# ── summary table  (matches LaTeX table layout) ───────────────────────────────
W = 145
print(f"\n\n{'=' * W}")
print(f"  SUMMARY   n_trials={N_TRIALS}")
print(f"{'=' * W}")
print(f"{'Strategy':<26} {'E_therm[N²m²s]':>16} {'E_mech[J]':>11} "
      f"{'P_peak[W]':>11} {'Err[deg]':>10} {'Jerk[N²m²/s]':>14} "
      f"{'FF[%]':>7} {'FB[%]':>7} {'UnifCost':>14} {'t_ctrl[s]':>10}")
print("-" * W)
for r in summary:
    print(
        f"{r['label']:<26} "
        f"{r['E_thermal']:>16.2f} "
        f"{r['E_mech']:>11.2f} "
        f"{r['P_peak']:>11.2f} "
        f"{r['tracking_err']:>10.4f} "
        f"{r['torque_jerk']:>14.2f} "
        f"{r['ff_pct']:>7.1f} "
        f"{r['fb_pct']:>7.1f} "
        f"{r['unified_cost']:>14.4e} "
        f"{r['opt_time_s']:>10.4f}"
    )
print("=" * W)

# ── Table 2: best found PD gains + alpha per strategy ─────────────────────────
print(f"\n\n{'=' * W}")
print(f"  BEST FOUND PARAMETERS PER STRATEGY")
print(f"{'=' * W}")
for r in summary:
    params_list = r["params_repr"]
    is_per_seg  = len(params_list) > 1
    print(f"\n  {r['label']}{'  (per-segment — showing each segment)' if is_per_seg else ''}:")
    header = f"  {'Seg':>4}  {'alpha':>6}  {'interp':<12}  {'kp (J1..J6)':<46}  kd (J1..J6)"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for seg_idx, p in enumerate(params_list, start=1):
        kp = np.array([p["kp1"], p["kp2"], p["kp3"], p["kp4"], p["kp5"], p["kp6"]])
        kd = np.array([p["kp1"]*p["ratio1"], p["kp2"]*p["ratio2"], p["kp3"]*p["ratio3"],
                       p["kp4"]*p["ratio4"], p["kp5"]*p["ratio5"], p["kp6"]*p["ratio6"]])
        kp_str = "[" + ", ".join(f"{v:7.2f}" for v in kp) + "]"
        kd_str = "[" + ", ".join(f"{v:6.3f}" for v in kd) + "]"
        interp = p.get("interpolation", "—")
        alpha  = p.get("alpha", float("nan"))
        seg_label = str(seg_idx) if is_per_seg else "all"
        print(f"  {seg_label:>4}  {alpha:>6.3f}  {interp:<12}  {kp_str:<46}  {kd_str}")
print(f"\n{'=' * W}")
print(f"\nPlots saved to: {OUT_DIR.resolve()}")
