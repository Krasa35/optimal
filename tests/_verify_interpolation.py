"""
Compare PD controller with different interpolation profiles (alpha=0.8, same kp/kd).
Mirrors the setup from examples/110_PDController_joint_alpha.ipynb cell 1.

For each interpolation method, this script produces:
  - plot_feedback_part  (FF vs total torque, closed-loop correction)
  - plot_error          (tracking error over time)
  - plot_metrics        (energy / power / jerk breakdown)
  - a summary table printed at the end
"""
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")           # headless – swap to "TkAgg" / "Qt5Agg" if you want interactive
import matplotlib.pyplot as plt

from optimal.RobotController import RobotController
from optimal.RobotMujocoModel import RobotMujocoModel
from optimal.RobotPinocchioModel import RobotPinocchioModel
from optimal.configuration import UR10_sandbox, Trajectories
from optimal.metrics import (
    thermal_energy, mechanical_work, peak_power,
    tracking_error, torque_jerk, plot_metrics, plot_controls_and_error, unified_cost
)

# ── setup (identical to notebook 110_ cell 1) ──────────────────────────────
cfg            = UR10_sandbox
robot_mj_model = RobotMujocoModel(cfg.mjcf_path, cfg.list_of_joints)
robot_pin_model = RobotPinocchioModel(cfg.urdf_path, cfg.list_of_joints)
controller     = RobotController(robot_pin_model)

DT   = robot_mj_model.model.opt.timestep
path = Trajectories.joint_6dof
T    = 1000
seg  = [T] * (len(path) - 1)
nq   = len(robot_pin_model.q_indices)

ALPHA          = 0.8
INTERPOLATIONS = ["linear", "cubic", "trapezoidal", "quintic"]

OUT_DIR = Path(__file__).parent / "interpolation_comparison"
OUT_DIR.mkdir(exist_ok=True)


@contextmanager
def _save_plots(prefix: str):
    """Wrap plt.show() so every open figure is saved before being shown."""
    counter = [0]
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

# ── helpers ─────────────────────────────────────────────────────────────────
def run_controller(interpolation: str):
    """Run open-loop planning then MuJoCo closed-loop simulation."""
    xs, us    = [], []
    v_start   = np.zeros(len(robot_pin_model.v_indices))

    t0 = time.time()
    for i in range(len(path) - 1):
        start  = np.concatenate([path[i],     v_start])
        target = np.concatenate([path[i + 1], np.zeros(nq)])
        t_xs, t_us = controller.compute_control(
            x_start=start, x_goal=target,
            T=T, DT=DT,
            kp=cfg.kp, kd=cfg.kd,
            alpha=ALPHA,
            interpolation=interpolation,
            option="pd",
        )
        xs.extend(t_xs)
        us.extend(t_us)
        v_start = xs[-1][nq:]
    controller_time = time.time() - t0

    robot_mj_model.reset()
    xs_real, us_real = robot_mj_model.simulate_control(
        us=us, xs=xs, kp=cfg.kp, kd=cfg.kd
    )
    return xs, us, xs_real, us_real, controller_time


def feedback_stats(us_real, us):
    """Return global FF coverage and FB correction percentages."""
    u_tot = np.asarray(us_real, dtype=float)
    u_ff  = np.asarray(us,      dtype=float)
    u_fb  = u_tot - u_ff
    g_tot = np.linalg.norm(u_tot)
    g_ff  = np.linalg.norm(u_ff)
    g_fb  = np.linalg.norm(u_fb)
    denom = g_tot if g_tot > 0 else 1.0
    return g_ff / denom * 100.0, g_fb / denom * 100.0


# ── main loop ───────────────────────────────────────────────────────────────
summary = []

for interp in INTERPOLATIONS:
    print(f"\n{'=' * 60}")
    print(f"  Interpolation: {interp}   (alpha={ALPHA})")
    print(f"{'=' * 60}")

    xs, us, xs_real, us_real, ctrl_time = run_controller(interp)

    # ── plots (also saved to OUT_DIR) ────────────────────────────────────
    print(f"\n[{interp}] controls + error:")
    with _save_plots(f"{interp}_controls_error"):
        plot_controls_and_error(us_real, xs_real, DT, path[1:], seg, title=f"PD Control — interpolation: {interp}  (α={ALPHA})")

    print(f"\n[{interp}] metrics:")
    with _save_plots(f"{interp}_metrics"):
        plot_metrics(us_real, xs_real, DT, path[1:], seg, controller_time=ctrl_time)

    # ── collect numbers ───────────────────────────────────────────────────
    us_arr  = np.asarray(us_real)
    xs_arr  = np.asarray(xs_real)

    E_th   = thermal_energy(us_arr, DT)
    E_mech = mechanical_work(us_arr, xs_arr, DT)
    P_pk   = peak_power(us_arr, xs_arr)
    t_err  = tracking_error(xs_arr, path[1:], seg)         # deg
    t_jerk = torque_jerk(us_arr)
    ff_pct, fb_pct = feedback_stats(us_real, us)
    cost = unified_cost(us_arr, xs_arr, DT, path[1:], seg)

    summary.append(dict(
        interpolation  = interp,
        E_thermal      = E_th,
        E_mech         = E_mech,
        P_peak         = P_pk,
        tracking_err   = t_err,
        torque_jerk    = t_jerk,
        ff_pct         = ff_pct,
        fb_pct         = fb_pct,
        unified_cost   = cost,
        ctrl_time_s    = ctrl_time,
    ))

# ── summary table ────────────────────────────────────────────────────────────
W = 120
print(f"\n\n{'=' * W}")
print(f"  SUMMARY   alpha={ALPHA}   kp={cfg.kp}   kd={cfg.kd}")
print(f"{'=' * W}")
print(f"{'Method':<14} {'E_therm[N²m²s]':>16} {'E_mech[J]':>12} "
      f"{'P_peak[W]':>11} {'Err[deg]':>10} {'Jerk[N²m²/s]':>14} "
      f"{'FF[%]':>7} {'FB[%]':>7} {'UnifCost':>14} {'t_ctrl[s]':>11}")
print("-" * W)
for r in summary:
    print(
        f"{r['interpolation']:<14} "
        f"{r['E_thermal']:>16.2f} "
        f"{r['E_mech']:>12.2f} "
        f"{r['P_peak']:>11.2f} "
        f"{r['tracking_err']:>10.4f} "
        f"{r['torque_jerk']:>14.2f} "
        f"{r['ff_pct']:>7.1f} "
        f"{r['fb_pct']:>7.1f} "
        f"{r['unified_cost']:>14.4e} "
        f"{r['ctrl_time_s']:>11.4f}"
    )
print("=" * W)
print(f"\nPlots saved to: {OUT_DIR.resolve()}")
