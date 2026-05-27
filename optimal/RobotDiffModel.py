import mujoco
import numpy as np
import crocoddyl
from .RobotMujocoModel import RobotMujocoModel

class RobotDiffActionModel(crocoddyl.DifferentialActionModelAbstract):
    def __init__(
        self,
        mjmodel: RobotMujocoModel,
        state: crocoddyl.StateVector,
        x_target: np.ndarray,
        w_q: float = 1.0,
        w_v: float = 1.0,
        w_u: float = 1.0,
        fd_eps: float = 1e-6,
    ) -> None:
        super().__init__(state, mjmodel.model.nu, 1)
        self.mjmodel = mjmodel
        self.mj_data = mujoco.MjData(mjmodel.model)
        self.q_indices = mjmodel.q_indices()
        self.v_indices = mjmodel.v_indices()
        self.qpos_nominal = mjmodel.data.qpos.copy()
        self.qvel_nominal = mjmodel.data.qvel.copy()
        self.x_target = np.asarray(x_target, dtype=float).copy()
        self.nq = self.q_indices.size
        self.nv = self.v_indices.size
        self.w_q = w_q
        self.w_v = w_v
        self.w_u = w_u
        self.fd_eps = fd_eps
        self.ctrl_min = self.mjmodel.model.actuator_ctrlrange[:, 0].copy() if self.mjmodel.model.nu > 0 else np.zeros(0)
        self.ctrl_max = self.mjmodel.model.actuator_ctrlrange[:, 1].copy() if self.mjmodel.model.nu > 0 else np.zeros(0)
        self.has_ctrl_limits = bool(np.any(self.mjmodel.model.actuator_ctrllimited)) if self.mjmodel.model.nu > 0 else False
        
        # Allocate temp data for finite-difference derivatives
        self.mj_data_plus = mujoco.MjData(mjmodel.model)
        self.mj_data_minus = mujoco.MjData(mjmodel.model)

    def calc(self, data, x, u=None) -> None:
        if u is None:
            u = np.zeros(self.nu)
        else:
            u = np.asarray(u, dtype=float).reshape(self.nu)

        if self.has_ctrl_limits:
            u = np.clip(u, self.ctrl_min, self.ctrl_max)

        q = x[: self.nq]
        v = x[self.nq :]

        # Reset the full state, then inject the reduced coordinates.
        self.mj_data.qpos[:] = self.qpos_nominal
        self.mj_data.qvel[:] = self.qvel_nominal
        self.mj_data.qpos[self.q_indices] = q
        self.mj_data.qvel[self.v_indices] = v
        if self.nu > 0:
            self.mj_data.ctrl[:] = u
        mujoco.mj_forward(self.mjmodel.model, self.mj_data)

        vdot = self.mj_data.qacc[self.v_indices].copy()
        data.xout[:] = vdot

        dq = q - self.x_target[: self.nq]
        dv = v - self.x_target[self.nq :]
        data.cost = (
            0.5 * self.w_q * float(dq @ dq)
            + 0.5 * self.w_v * float(dv @ dv)
            + 0.5 * self.w_u * float(u @ u)
        )