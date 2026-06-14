from pathlib import Path
import logging
from pathlib import Path
from xml.parsers.expat import model
import mujoco
import numpy as np
import matplotlib.pyplot as plt

class RobotMujocoModel:
    def __init__(self, model_path: str):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.joint_names = self._joint_names()


        logging.info(f"Model loaded from {model_path}.")

    def _joint_names(self) -> list[str]:
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]

    @property
    def joint_positions(self) -> dict[str, float]:
        q: dict[str, float] = {}
        for joint_name in self.actuated_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                q[joint_name] = self.data.qpos[self.model.jnt_qposadr[joint_id]]
        return q

    @joint_positions.setter
    def joint_positions(self, q) -> None:
        if type(q) is list:
            q = dict(zip(self.actuated_joint_names, q))
        else:
            if not isinstance(q, dict):
                raise TypeError("Input must be a list or a dictionary")
        for joint_name, target_pos in q.items():
                try:
                    # Find the ID of the joint
                    joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    if joint_id != -1:
                        # Find where this joint's data lives in the qpos array
                        qpos_adr = self.model.jnt_qposadr[joint_id]
                        
                        # Update both the current position and the controller target
                        self.data.qpos[qpos_adr] = target_pos
                        
                        # If you have actuators with the same name, set their targets too
                        actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
                        if actuator_id != -1:
                            self.data.ctrl[actuator_id] = target_pos
                        mujoco.mj_forward(self.model, self.data) 
                except Exception as e:
                    print(f"Error setting {joint_name}: {e}")

    @property
    def actuated_joint_positions(self) -> dict[str, float]:
        q: dict[str, float] = {}
        for joint_name in self.actuated_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                q[joint_name] = self.data.qpos[self.model.jnt_qposadr[joint_id]]
        return q

    @property
    def actuated_joint_names(self) -> list[str]:
        actuated_joint_names = []
        for i in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[i, 0])
            if joint_id < 0:
                continue
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name is not None:
                actuated_joint_names.append(joint_name)
        return actuated_joint_names
    
    @actuated_joint_names.setter
    def actuated_joint_names(self, joint_names: list[str]) -> None:
        for i in range(self.model.nu):
            if i < len(joint_names):
                joint_name = joint_names[i]
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id != -1:
                    self.model.actuator_trnid[i, 0] = joint_id
                else:
                    print(f"Warning: Joint '{joint_name}' not found in the model.")
            else:
                self.model.actuator_trnid[i, 0] = -1

    def q_indices(self, list_of_joints: list[str]=[]) -> np.ndarray:
        if not list_of_joints:
            list_of_joints = self.actuated_joint_names  
        return np.array([self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)] for joint_name in list_of_joints], dtype=int)
    
    def v_indices(self, list_of_joints: list[str]=[]) -> np.ndarray:
        if not list_of_joints:
            list_of_joints = self.actuated_joint_names
        return np.array([self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)] for joint_name in list_of_joints], dtype=int)
    
    def current_state(self) -> np.ndarray:
        q_indices = self.q_indices()
        v_indices = self.v_indices()
        return np.concatenate([self.data.qpos[q_indices], self.data.qvel[v_indices]])

    def random_actuated_q(
        self,
        default_range: tuple[float, float] = (-np.pi, np.pi),
        seed: int | None = None,
        ensure_feasible: bool = False,
        max_tries: int = 200,
        constraint_tol: float = 1e-2,
    ) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        joint_names: list[str] = []
        joint_ids: list[int] = []
        qpos_addrs: list[int] = []
        limits: list[tuple[float, float]] = []
        for joint_name in self.actuated_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                continue

            if int(self.model.jnt_limited[joint_id]) == 1:
                low, high = self.model.jnt_range[joint_id]
            else:
                low, high = default_range

            joint_names.append(joint_name)
            joint_ids.append(joint_id)
            qpos_addrs.append(int(self.model.jnt_qposadr[joint_id]))
            limits.append((float(low), float(high)))

        if not joint_names:
            return {}

        if not ensure_feasible:
            return {
                name: float(rng.uniform(low, high))
                for name, (low, high) in zip(joint_names, limits)
            }

        # if not hasattr(mujoco, "mj_projectConstraints"):
            # raise RuntimeError("mujoco.mj_projectConstraints is required for ensure_feasible=True.")

        qpos_addrs_arr = np.asarray(qpos_addrs, dtype=int)
        qpos_nominal = self.data.qpos.copy()
        qvel_nominal = self.data.qvel.copy()
        ctrl_nominal = self.data.ctrl.copy()
        try:
            for _ in range(max_tries):
                q_sample = np.array(
                    [rng.uniform(low, high) for (low, high) in limits],
                    dtype=float,
                )
                self.data.qpos[:] = qpos_nominal
                self.data.qvel[:] = qvel_nominal
                self.data.ctrl[:] = ctrl_nominal
                self.data.qpos[qpos_addrs_arr] = q_sample
                mujoco.mj_forward(self.model, self.data)
                mujoco.mj_projectConstraint(self.model, self.data)
                mujoco.mj_forward(self.model, self.data)

                if self.data.nefc > 0 and np.max(np.abs(self.data.efc_pos)) > constraint_tol:
                    continue
                if self.data.ncon > 0:
                    continue

                q_projected = self.data.qpos[qpos_addrs_arr].copy()
                in_range = True
                for (low, high), val in zip(limits, q_projected):
                    if val < low - constraint_tol or val > high + constraint_tol:
                        in_range = False
                        break
                if not in_range:
                    continue

                return {
                    name: float(val)
                    for name, val in zip(joint_names, q_projected)
                }
        finally:
            self.data.qpos[:] = qpos_nominal
            self.data.qvel[:] = qvel_nominal
            self.data.ctrl[:] = ctrl_nominal
            mujoco.mj_forward(self.model, self.data)

        raise RuntimeError(
            "Unable to sample a feasible actuated configuration within max_tries."
        )

    def reset(self) -> None:
        self.data.qpos[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data) 

    def snapshot(self, plot: bool = False) -> np.ndarray:
        renderer = mujoco.Renderer(self.model, height=480, width=640)
        renderer.update_scene(self.data)
        if plot:
            plt.imshow(renderer.render())
            plt.axis('off')
            plt.show()
        else:
            return renderer.render()

if __name__ == "__main__":
    MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "UR5gripper_2_finger_KKI.xml"
    model = RobotMujocoModel(MODEL_PATH)
    print("Joint names:", model.joint_names)
    print("Initial joint positions:", model.joint_positions)
    print("Actuated joint names:", model.actuated_joint_names)
    print("Initial q indices:", model.q_indices())
    print("Initial v indices:", model.v_indices())
    model.actuated_joint_names = model.joint_names[:4]  # Assuming the first 4 joints are actuated
    print("Updated actuated joint names:", model.actuated_joint_names)
    print("Updated q indices:", model.q_indices())
    print("Updated v indices:", model.v_indices())

    # Example of setting joint positions
    target_positions = {
        "shoulder_pan_joint": 0.5,
        "shoulder_lift_joint": -0.5,
        "elbow_joint": 0.5,
        "wrist_1_joint": -0.5,
        "wrist_2_joint": 0.5,
        "wrist_3_joint": -0.5
    }
    model.joint_positions = target_positions
    print("Updated joint positions:", model.joint_positions)
