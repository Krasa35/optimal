import numpy as np
import crocoddyl
import mujoco

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


def precision(xs, targets, seg_lengths, nq):
    xs = np.asarray(xs)
    idx = 0
    errs = []
    for tgt, n in zip(targets, seg_lengths):
        idx += n
        errs.append(terminal_error(xs[idx - 1, :nq], np.asarray(tgt)[:nq]))
    return float(np.mean(errs)) * 180 / np.pi


# Headless replica of mpc_visualize (no viewer) to verify MPC precision/energy.
DT_ocp = 0.01
T_mpc = 100
step = int(round(DT_ocp / DT))
N_steps = 1000
seg = [N_steps] * (len(path) - 1)

func = lambda warm_xs, warm_us, x_start, x_target: controller.compute_control(
    x_start, x_target, T=T_mpc, DT=DT_ocp,
    option="crocoddyl_advanced", solver=crocoddyl.SolverBoxFDDP,
    kp=cfg.kp, kd=cfg.kd, warm_xs=warm_xs, warm_us=warm_us)

mj.reset()
qi, vi, ui = mj.q_indices, mj.v_indices, mj.u_indices
x_real, u_real = [], []
warm_xs = warm_us = x_computed = None
u_computed = np.zeros((1, len(ui)))
for i in range(len(path) - 1):
    for s in range(seg[i]):
        if s % step == 0:
            q_c = mj.data.qpos[qi].copy()
            v_c = mj.data.qvel[vi].copy()
            x_c = np.concatenate([q_c, v_c])
            x_des = np.concatenate([np.asarray(path[i + 1]), np.zeros(len(vi))])
            if x_computed is not None and len(u_computed) >= step:
                warm_xs = np.concatenate([[x_c], x_computed[step:], np.tile(x_computed[-1:], (step, 1))])
                warm_us = np.concatenate([u_computed[step:], np.tile(u_computed[-1:], (step, 1))])
            else:
                warm_xs = warm_us = None
            x_computed, u_computed = func(warm_xs, warm_us, x_c, x_des)
            u_computed = np.clip(np.asarray(u_computed, float),
                                 mj.model.actuator_ctrlrange[ui, 0],
                                 mj.model.actuator_ctrlrange[ui, 1])
        mj.data.ctrl[ui] = u_computed[0]
        mujoco.mj_step(mj.model, mj.data)
        x_real.append(np.concatenate([mj.data.qpos[qi], mj.data.qvel[vi]]))
        u_real.append(mj.data.ctrl[ui].copy())

xs_mpc = np.array(x_real)
us_mpc = np.array(u_real)
print("MPC (crocoddyl_advanced):")
print("  E_thermal :", round(thermal_energy(us_mpc, DT), 2))
print("  E_mech    :", round(mechanical_work(us_mpc, xs_mpc, DT), 2))
print("  P_peak    :", round(peak_power(us_mpc, xs_mpc), 2))
print("  precision :", round(precision(xs_mpc, path[1:], seg, nq), 4), "deg")
