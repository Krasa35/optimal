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
        kp: float = 3.0,
        kd: float = 0.2,
        torque_limits: float | np.ndarray | None = None,
    ):
        self.model = model
        self.data = model.createData()
        if len(kp) == 1:
            self.kp = np.full(model.nq, float(kp), dtype=float)
        else:
            self.kp = np.asarray(kp, dtype=float)
        if len(kd) == 1:
            self.kd = np.full(model.nq, float(kd), dtype=float)
        else:
            self.kd = np.asarray(kd, dtype=float)
        
        self.nq = model.nq
        self.nv = model.nv
        self.nx = self.nq + self.nv
        self.has_ctrl_limits = True if torque_limits is not None else False
        self.ctrl_min = -torque_limits if torque_limits is not None else None
        self.ctrl_max = torque_limits if torque_limits is not None else None

    def compute_control(
        self,
        x_start: np.ndarray,
        x_goal: np.ndarray,
        horizon: int,
        control_dt: float = 1.0,
        control_noise_std: float = 0.0,
        alpha: float = 0.0,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        x_start = np.asarray(x_start, dtype=float)
        x_goal = np.asarray(x_goal, dtype=float)
    
        if x_start.shape == (self.nq,):
            x_start = np.concatenate([x_start, np.zeros(self.nv)])
        if x_goal.shape == (self.nq,):
            x_goal = np.concatenate([x_goal, np.zeros(self.nv)])
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
        q_start = x_start[: self.nq].copy()
        v_target = x_goal[self.nq :]

        e_prev = q_target - q
        xs = [x_start.copy()]
        us = []

        for step in range(horizon):
            if alpha > 0.0:
                _alpha = min(1.0, step / (horizon * alpha))
                q_ref = q_start + _alpha * (q_target - q_start)
                err = q_ref - q
            else:
                err = q_target - q
            derr = (err - e_prev) / control_dt
            e_prev = err
            
            # Obliczenie momentów
            u = self.kp * err + self.kd * derr
            g = pin.computeGeneralizedGravity(self.model, self.data, q)
            tau = u + g
            
            # 3. NASYCENIE (TORQUE CLIPPING)
            # Odcina wszystkie kosmiczne wartości do bezpiecznych limitów robota
            tau = np.clip(tau, -self.ctrl_max, self.ctrl_max)
            
            # Krok wirtualnej fizyki Pinocchio
            a = pin.aba(self.model, self.data, q, v, tau)
            v = v + control_dt * a
            q = pin.integrate(self.model, q, control_dt * v)
            
            us.append(tau.copy())
            xs.append(np.concatenate([q.copy(), v.copy()]))
            
        return xs, us