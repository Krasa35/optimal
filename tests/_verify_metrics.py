import numpy as np
import crocoddyl

from optimal.RobotController import RobotController
from optimal.RobotMujocoModel import RobotMujocoModel
from optimal.RobotPinocchioModel import RobotPinocchioModel
from optimal.configuration import UR10_sandbox, Trajectories
from optimal.metrics import thermal_energy, mechanical_work, peak_power, terminal_error

cfg = UR10_sandbox
mj = RobotMujocoModel(cfg.mjcf_path, cfg.list_of_joints)
pinm = RobotPinocchioModel(cfg.urdf_path, cfg.list_of_joints)
controller = RobotController(pinm)
DT = mj.model.opt.timestep
path = Trajectories.joint_6dof
nq = len(pinm.q_indices)
T = 1000
seg = [T] * (len(path) - 1)


def precision(xs, targets, seg_lengths, nq):
    xs = np.asarray(xs)
    idx = 0
    errs = []
    for tgt, n in zip(targets, seg_lengths):
        idx += n
        errs.append(terminal_error(xs[idx - 1, :nq], np.asarray(tgt)[:nq]))
    return float(np.mean(errs))


def run(option, **kw):
    xs, us = [], []
    v_start = np.zeros(len(pinm.v_indices))
    for i in range(len(path) - 1):
        start = np.concatenate([path[i], v_start])
        target = np.concatenate([path[i + 1], np.zeros(len(pinm.v_indices))])
        t_xs, t_us = controller.compute_control(
            x_start=start, x_goal=target, T=T, DT=DT,
            kp=cfg.kp, kd=cfg.kd, option=option, **kw)
        xs.extend(t_xs)
        us.extend(t_us)
        v_start = xs[-1][nq:]
    mj.reset()
    xs_real, us_real = mj.simulate_control(us=us, xs=xs, kp=cfg.kp, kd=cfg.kd)
    return xs_real, us_real


configs = [
    ("PD (no alpha)", "pd", {}),
    ("PD (alpha=0.5)", "pd", {"alpha": 0.5}),
    ("Crocoddyl basic", "crocoddyl", {"solver": crocoddyl.SolverBoxFDDP}),
    ("Crocoddyl advanced", "crocoddyl_advanced", {"solver": crocoddyl.SolverBoxFDDP}),
]

print(f"{'method':<22}{'E_thermal':>12}{'E_mech':>12}{'P_peak':>12}{'precision[deg]':>16}")
for name, opt, kw in configs:
    xs_real, us_real = run(opt, **kw)
    E_th = thermal_energy(us_real, DT)
    E_mech = mechanical_work(us_real, xs_real, DT)
    P_pk = peak_power(us_real, xs_real)
    prec = precision(xs_real, path[1:], seg, nq) * 180 / np.pi
    print(f"{name:<22}{E_th:>12.2f}{E_mech:>12.2f}{P_pk:>12.2f}{prec:>16.4f}")
