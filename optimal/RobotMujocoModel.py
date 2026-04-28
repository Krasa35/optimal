import logging
from xml.parsers.expat import model
import mujoco
import numpy as np

class RobotMujocoModel:
    def __init__(self, model_path: str):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        logging.info(f"Model loaded from {model_path}.")

    def set_joint_values(self, q: dict[str, float]) -> None:
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
            except Exception as e:
                print(f"Error setting {joint_name}: {e}")

    def reset(self) -> None:
        self.data.qpos[:] = 0.0
        self.data.ctrl[:] = 0.0

    def snapshot(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data) 
        renderer = mujoco.Renderer(self.model, height=480, width=640)
        renderer.update_scene(self.data)
        return renderer.render()
    
    @property
    def joint_names(self) -> list[str]:
        return [self.model.joint(joint_id).name for joint_id in range(self.model.njnt)]
    
    @property
    def joint_positions(self) -> list[str]:
        return self.data.qpos[:self.model.njnt]
        