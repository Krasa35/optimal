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
        alpha: float = 0.0,
    ):
        self.model = model
        self.data = model.createData()
        if isinstance(kp, (int, float)):
            self.kp = np.full(model.nq, float(kp), dtype=float)
        elif len(kp) == 1:
            self.kp = np.full(model.nq, float(kp[0]), dtype=float)
        else:
            self.kp = np.asarray(kp, dtype=float)
            
        if isinstance(kd, (int, float)):
            self.kd = np.full(model.nq, float(kd), dtype=float)
        elif len(kd) == 1:
            self.kd = np.full(model.nq, float(kd[0]), dtype=float)
        else:
            self.kd = np.asarray(kd, dtype=float)
        
        self.nq = model.nq
        self.nv = model.nv
        self.nx = self.nq + self.nv
        self.ctrl_min = -model.effortLimit
        self.ctrl_max = model.effortLimit
        self.alpha = float(alpha)

    def compute_control(
        self,
        x_start: np.ndarray,
        x_goal: np.ndarray,
        horizon: int,
        control_dt: float = 1.0,
        control_noise_std: float = 0.0,
        alpha: float = 0.0,
        interpolation: str = "quintic",
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

        if alpha == 0.0:
            alpha = self.alpha

        if alpha > 0.0:
            e_prev = q_start - q
        else:
            e_prev = q_target - q

        xs = [x_start.copy()]
        us = []

        for step in range(horizon):
            if alpha > 0.0:
                t = min(1.0, step / (horizon * alpha))
                
                if interpolation == "linear":
                    _alpha = t
                elif interpolation == "cubic":
                    _alpha = 3 * t**2 - 2 * t**3
                elif interpolation == "quintic":
                    _alpha = 10 * t**3 - 15 * t**4 + 6 * t**5
                elif interpolation == "trapezoidal":
                    blend_time = 0.2
                    if t <= blend_time:
                        _alpha = (1 / (2 * blend_time * (1 - blend_time))) * t**2
                    elif t <= (1 - blend_time):
                        _alpha = (1 / (1 - blend_time)) * (t - (blend_time / 2))
                    else:
                        _alpha = 1 - (1 / (2 * blend_time * (1 - blend_time))) * (1 - t)**2
                else:
                    raise ValueError(f"Unknown interpolation type: {interpolation}")

                q_ref = q_start + _alpha * (q_target - q_start)
                err = q_ref - q
            else:
                err = q_target - q
            derr = (err - e_prev) / control_dt
            e_prev = err
            
            u = self.kp * err + self.kd * derr
            g = pin.computeGeneralizedGravity(self.model, self.data, q)
            tau = u + g
            
            tau = np.clip(tau, -self.ctrl_max, self.ctrl_max)
            
            a = pin.aba(self.model, self.data, q, v, tau)
            v = v + control_dt * a
            q = pin.integrate(self.model, q, control_dt * v)
            
            us.append(tau.copy())
            xs.append(np.concatenate([q.copy(), v.copy()]))
            
        return xs, us