import mujoco
import numpy as np
import crocoddyl

class RobotDiffModel(crocoddyl.DifferentialActionModelAbstract):
    def __init__(
        self,
        model: mujoco.MjModel,
        q_indices: np.ndarray,
        v_indices: np.ndarray,
        q_nominal: np.ndarray,
        v_nominal: np.ndarray,
        state: crocoddyl.StateVector,
        x_target: np.ndarray,
        w_q: float,
        w_v: float,
        w_u: float,
    ) -> None:
        super().__init__(state, model.nu, 1)
        self.model = model
        self.mj_data = mujoco.MjData(model)
        self.q_indices = q_indices
        self.v_indices = v_indices
        self.q_nominal = q_nominal.copy()
        self.v_nominal = v_nominal.copy()
        self.x_target = x_target.copy()
        self.nq = q_indices.sizeMae
        self.nv = v_indices.size
        self.w_q = w_q
        self.w_v = w_v
        self.w_u = w_u
        self.ctrl_min = model.actuator_ctrlrange[:, 0].copy() if model.nu > 0 else np.zeros(0)
        self.ctrl_max = model.actuator_ctrlrange[:, 1].copy() if model.nu > 0 else np.zeros(0)
        self.has_ctrl_limits = bool(np.any(model.actuator_ctrllimited)) if model.nu > 0 else False

    def calc(self, data, x, u=None) -> None:
        if u is None:
            u = np.zeros(self.nu)
        else:
            u = np.asarray(u, dtype=float).reshape(self.nu)

        if self.has_ctrl_limits:
            u = np.clip(u, self.ctrl_min, self.ctrl_max)

        q = x[: self.nq]
        v = x[self.nq :]

        self.mj_data.qpos[:] = self.q_nominal
        self.mj_data.qvel[:] = self.v_nominal
        self.mj_data.qpos[self.q_indices] = q
        self.mj_data.qvel[self.v_indices] = v
        if self.model.nu > 0:
            self.mj_data.ctrl[:] = u
        mujoco.mj_forward(self.model, self.mj_data)

        vdot = self.mj_data.qacc[self.v_indices].copy()
        data.xout[:] = vdot

        dq = q - self.x_target[: self.nq]
        dv = v - self.x_target[self.nq :]
        data.cost = (
            0.5 * self.w_q * float(dq @ dq)
            + 0.5 * self.w_v * float(dv @ dv)
            + 0.5 * self.w_u * float(u @ u)
        )

    def calcDiff(self, data, x, u=None) -> None:
        pass
