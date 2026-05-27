from pathlib import Path
import mujoco
import numpy as np

try:
    from optimal.RobotMujocoModel import RobotMujocoModel
except ImportError:  # Allows running this file directly as a script.
    from RobotMujocoModel import RobotMujocoModel


class RobotPDController:
    def __init__(self, model: RobotMujocoModel, kp: float = 3.0, kd: float = .2):
        self.mjmodel = model
        self.model = model.model
        self.data = model.data
        self.kp = kp
        self.kd = kd
        self.nu = model.model.nu
        self.q_indices = model.q_indices()
        self.v_indices = model.v_indices()
        self.joint_names = model.actuated_joint_names
        self.joint_to_actuator = {name: idx for idx, name in enumerate(self.joint_names)}
        self.nx = self.q_indices.size + self.v_indices.size
        self.ctrl_min = self.model.actuator_ctrlrange[:, 0] if self.nu > 0 else np.zeros(0)
        self.ctrl_max = self.model.actuator_ctrlrange[:, 1] if self.nu > 0 else np.zeros(0)
        self.has_ctrl_limits = bool(np.any(self.model.actuator_ctrllimited)) if self.nu > 0 else False

    def compute_control(
        self, x_start: np.ndarray, x_goal: np.ndarray, horizon: int, control_dt: float = 1.0, control_noise_std: float = 0.0
    ) -> np.ndarray:
        assert x_start.shape == (self.nx,)
        assert x_goal.shape == (self.nx,)
        if self.nu != self.v_indices.size:
            raise ValueError("Actuator count does not match the number of actuated DoFs.")
        if control_dt <= 0:
            raise ValueError("control_dt must be positive.")
        if control_noise_std < 0:
            raise ValueError("control_noise_std cannot be negative.")
        
        q0 = x_start[: self.q_indices.size]
        v0 = x_start[self.q_indices.size :]
        q_target = x_goal[: self.q_indices.size]
        mj_dt = float(self.model.opt.timestep)
        n_substeps = max(1, int(round(control_dt / mj_dt)))
        effective_dt = n_substeps * mj_dt
        rng = np.random.default_rng()
        warm_xs = [x_start.copy()]
        warm_us = []
        e_prev = None

        self.data.qpos[self.q_indices] = q0
        self.data.qvel[self.v_indices] = v0
        mujoco.mj_forward(self.model, self.data)

        for k in range(horizon):
            # alpha = (k + 1) / horizon
            # q_des = (1.0 - alpha) * q0 + alpha * q_target
            q_curr = self.data.qpos[self.q_indices]
            e = q_target - q_curr
            if e_prev is None:
                e_prev = e.copy()
            de = (e - e_prev) / effective_dt
            e_prev = e
            # tau_local = .01*self.data.qfrc_bias[self.v_indices] + self.kp * e + self.kd * de
            tau_local = self.kp * e + self.kd * de

            u = tau_local.copy()
            if self.has_ctrl_limits:
                u = np.clip(u, self.ctrl_min, self.ctrl_max)
            if control_noise_std > 0:
                u += rng.normal(scale=control_noise_std, size=u.shape)
            for _ in range(n_substeps):
                self.data.ctrl[:] = u
                mujoco.mj_step(self.model, self.data)
            warm_us.append(u.copy())
            warm_xs.append(np.concatenate([self.data.qpos[self.q_indices], self.data.qvel[self.v_indices]]))

        self.mjmodel.reset()  

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
    v_indices = model.v_indices()
    x_start = model.current_state()
    q_target = np.array(list(model.random_actuated_q().values()), dtype=float)
    x_goal = np.concatenate([q_target, np.zeros_like(v_indices, dtype=float)])
    print("Target state (x_goal):", x_goal)
    warm_xs, warm_us = controller.compute_control(x_start, x_goal, horizon=200)
    print("Warm-start controls:", warm_us[:5])
    
