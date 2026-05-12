from pathlib import Path
import mujoco
import numpy as np

try:
    from optimal.RobotMujocoModel import RobotMujocoModel
except ImportError:  # Allows running this file directly as a script.
    from RobotMujocoModel import RobotMujocoModel


class RobotPDController:
    def __init__(self, model: RobotMujocoModel, kp: float = 100.0, kd: float = 20.0):
        self.model = model
        self.kp = kp
        self.kd = kd
        self.nu = model.model.nu
        self.q_indices = model.q_indices()
        self.v_indices = model.v_indices()
        self.nx = self.q_indices.size + self.v_indices.size

        self.joint_to_actuator = np.full(model.q_indices().size, -1, dtype=int)
        for i in range(self.nu):
            joint_id = int(model.model.actuator_trnid[i, 0])
            if joint_id >= 0:
                local_joint_idx = np.where(self.q_indices == model.model.jnt_qposadr[joint_id])[0]
                if local_joint_idx.size > 0:
                    self.joint_to_actuator[local_joint_idx[0]] = i
                else:
                    joint_name = mujoco.mj_id2name(model.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                    print(
                        f"Warning: Actuator {i} is mapped to joint '{joint_name}' "
                        "which is not in the q_indices list."
                    )
            else:
                print(f"Warning: Actuator {i} is not mapped to any joint.")

    def compute_control(self, x: np.ndarray, x_target: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(self.nx)
        x_target = np.asarray(x_target, dtype=float).reshape(self.nx)

        q = x[: self.q_indices.size]
        v = x[self.q_indices.size :]
        q_des = x_target[: self.q_indices.size]
        v_des = x_target[self.q_indices.size :]
        tau_local = self.kp * (q_des - q) + self.kd * (v_des - v)

        u = np.zeros(self.nu)
        for local_idx, actuator_id in enumerate(self.joint_to_actuator):
            if actuator_id >= 0:
                u[actuator_id] = tau_local[local_idx]
        return u


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
    
