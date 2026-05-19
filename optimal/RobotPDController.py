from pathlib import Path
import mujoco
import numpy as np

try:
    from optimal.RobotMujocoModel import RobotMujocoModel
except ImportError:  # Allows running this file directly as a script.
    from RobotMujocoModel import RobotMujocoModel


class RobotPDController:
    def __init__(self, model: RobotMujocoModel, kp: float = 100.0, kd: float = 20.0):
        self.model = model.model
        self.data = model.data
        self.kp = kp
        self.kd = kd
        self.nu = model.model.nu
        self.q_indices = model.q_indices()
        self.v_indices = model.v_indices()
        self.joint_names = model.actuated_joint_names
        self.nx = self.q_indices.size + self.v_indices.size
        self.ctrl_min = self.model.actuator_ctrlrange[:, 0] if self.nu > 0 else np.zeros(0)
        self.ctrl_max = self.model.actuator_ctrlrange[:, 1] if self.nu > 0 else np.zeros(0)
        self.has_ctrl_limits = bool(np.any(self.model.actuator_ctrllimited)) if self.nu > 0 else False

    def compute_control(self, x_start: np.ndarray, x_goal: np.ndarray, horizon: int) -> np.ndarray:
        assert x_start.shape == (self.nx,)
        assert x_goal.shape == (self.nx,)

        q0 = x_start[: self.q_indices.size]
        v0 = x_start[self.q_indices.size :]
        q_target = x_goal[: self.q_indices.size]
        v_target = x_goal[self.q_indices.size :]
        warm_xs = [x_start.copy()]
        warm_us = []

        for k in range(horizon):
            alpha = (k + 1) / horizon
            q_des = (1.0 - alpha) * q0 + alpha * q_target
            q_curr = self.data.qpos[self.q_indices]
            v_curr = self.data.qvel[self.v_indices]
            tau_local = self.kp * (q_des - q_curr) + self.kd * (v_target - v_curr)

            u = np.zeros(self.nu)
            for local_idx, actuator_id in zip(self.q_indices, self.v_indices):
                u[local_idx] = tau_local[actuator_id]
            if self.has_ctrl_limits:
                u = np.clip(u, self.ctrl_min, self.ctrl_max)
            self.data.ctrl[:] = u
            mujoco.mj_step(self.model, self.data)
            warm_us.append(u.copy())
            warm_xs.append(np.concatenate([self.data.qpos[self.q_indices], self.data.qvel[self.v_indices]]))

        return warm_xs, warm_us


if __name__ == "__main__":
    MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "UR5gripper_2_finger_KKI.xml"
    model = RobotMujocoModel(MODEL_PATH)
    print("Joint names:", model.joint_names)
    print("Initial joint positions:", model.joint_positions)
    print("Actuated joint names:", model.actuated_joint_names)
    controller = RobotPDController(model)
    print("Joint to actuator mapping:", controller.joint_to_actuator)
    # Example of computing control
    q_indices = model.q_indices()
    v_indices = model.v_indices()
    x = np.concatenate([list(model.random_actuated_q().values()), np.zeros_like(v_indices)])
    print("Target state (x):", x)
    u = controller.compute_control(x)
    print("Computed control:", u)
    
