from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pinocchio as pin


class PDController:
    def __init__(
        self,
        model: pin.Model,
        *,
        controlled_joint_names: Iterable[str] | None = None,
        kp: float = 3.0,
        kd: float = 0.2,
        torque_limits: float | np.ndarray | None = None,
    ):
        self.model = model
        self.data = model.createData()
        self.kp = float(kp)
        self.kd = float(kd)
        self.nq = model.nq
        self.nv = model.nv
        self.nx = self.nq + self.nv

        self.controlled_v_indices = self._build_controlled_v_indices(controlled_joint_names)
        self.nu = int(self.controlled_v_indices.size)
        if self.nu == 0:
            raise ValueError("No controlled DoFs selected.")

        self.ctrl_min, self.ctrl_max, self.has_ctrl_limits = self._build_torque_limits(torque_limits)

    def _build_controlled_v_indices(self, controlled_joint_names: Iterable[str] | None) -> np.ndarray:
        if controlled_joint_names is None:
            return np.arange(self.nv, dtype=int)

        names = set(self.model.names.tolist())
        v_indices: list[int] = []
        for joint_name in controlled_joint_names:
            if joint_name not in names:
                raise ValueError(f"Joint '{joint_name}' is not present in the Pinocchio model.")
            joint_id = self.model.getJointId(joint_name)
            joint = self.model.joints[joint_id]
            if joint.nv <= 0:
                continue
            v_indices.extend(range(joint.idx_v, joint.idx_v + joint.nv))

        unique_sorted = np.array(sorted(set(v_indices)), dtype=int)
        if unique_sorted.size == 0:
            raise ValueError("controlled_joint_names does not include any movable joints.")
        return unique_sorted

    def _build_torque_limits(
        self, torque_limits: float | np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        if torque_limits is None:
            zeros = np.zeros(self.nu, dtype=float)
            return zeros, zeros, False

        limits = np.asarray(torque_limits, dtype=float)
        if limits.ndim == 0:
            if limits < 0:
                raise ValueError("torque_limits must be non-negative.")
            max_abs = np.full(self.nu, float(limits), dtype=float)
        elif limits.shape == (self.nu,):
            if np.any(limits < 0):
                raise ValueError("torque_limits must be non-negative.")
            max_abs = limits
        else:
            raise ValueError(f"torque_limits must be scalar or shape ({self.nu},), got {limits.shape}.")

        return -max_abs, max_abs, True

    def compute_control(
        self,
        x_start: np.ndarray,
        x_goal: np.ndarray,
        horizon: int,
        control_dt: float = 1.0,
        control_noise_std: float = 0.0,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        x_start = np.asarray(x_start, dtype=float)
        x_goal = np.asarray(x_goal, dtype=float)

        if x_start.shape != (self.nx,):
            raise ValueError(f"x_start must have shape ({self.nx},), got {x_start.shape}.")
        if x_goal.shape != (self.nx,):
            raise ValueError(f"x_goal must have shape ({self.nx},), got {x_goal.shape}.")
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        if control_dt <= 0:
            raise ValueError("control_dt must be positive.")
        if control_noise_std < 0:
            raise ValueError("control_noise_std cannot be negative.")

        q = x_start[: self.nq].copy()
        v = x_start[self.nq :].copy()
        q_target = x_goal[: self.nq]
        v_target = x_goal[self.nq :]

        rng = np.random.default_rng()
        warm_xs: list[np.ndarray] = [x_start.copy()]
        warm_us: list[np.ndarray] = []

        for _ in range(horizon):
            q_err_tangent = pin.difference(self.model, q, q_target)
            v_err = v_target - v

            u = self.kp * q_err_tangent[self.controlled_v_indices] + self.kd * v_err[self.controlled_v_indices]
            if control_noise_std > 0:
                u += rng.normal(scale=control_noise_std, size=u.shape)
            if self.has_ctrl_limits:
                u = np.clip(u, self.ctrl_min, self.ctrl_max)

            tau_full = np.zeros(self.nv, dtype=float)
            tau_full[self.controlled_v_indices] = u

            a = pin.aba(self.model, self.data, q, v, tau_full)
            v = v + control_dt * a
            q = pin.integrate(self.model, q, control_dt * v)

            warm_us.append(u.copy())
            warm_xs.append(np.concatenate([q.copy(), v.copy()]))

        return warm_xs, warm_us


# Backward-compatible name used in older scripts.
RobotPDController = PDController


if __name__ == "__main__":
    MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "ur5e" / "ur5e.urdf"
    pin_model = pin.buildModelFromUrdf(str(MODEL_PATH))
    controller = PDController(pin_model)
    x0 = np.zeros(pin_model.nq + pin_model.nv, dtype=float)
    xg = x0.copy()
    xg[: pin_model.nq] = 0.2
    xs, us = controller.compute_control(x0, xg, horizon=10, control_dt=0.01)
    print(f"Generated {len(xs)} states and {len(us)} controls.")
