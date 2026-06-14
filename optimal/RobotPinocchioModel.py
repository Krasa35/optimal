import numpy as np
import pinocchio as pin


class RobotPinocchioModel:
    def __init__(self, urdf_path, list_of_joints=None, start_position=None):
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()

        if list_of_joints is None:
            list_of_joints = self._all_joint_names()
        assert isinstance(list_of_joints, list), "list_of_joints should be a list of joint names."
        self.joint_names = list_of_joints
        self.q_indices = self._q_indices(list_of_joints)
        self.v_indices = self._v_indices(list_of_joints)
        self.start_position = start_position if start_position is not None else np.zeros(len(self.q_indices))
        self.reset()

    def _all_joint_names(self) -> list[str]:
        # Skip 'universe' (index 0) — it is the fixed world joint
        return [self.model.names[i] for i in range(1, self.model.njoints)]

    def _q_indices(self, list_of_joints=None) -> np.ndarray:
        if list_of_joints is None:
            list_of_joints = self.joint_names
        return np.array([
            self.model.joints[self.model.getJointId(name)].idx_q
            for name in list_of_joints
        ], dtype=int)

    def _v_indices(self, list_of_joints=None) -> np.ndarray:
        if list_of_joints is None:
            list_of_joints = self.joint_names
        return np.array([
            self.model.joints[self.model.getJointId(name)].idx_v
            for name in list_of_joints
        ], dtype=int)
    
    def _build_reduced_model(self) -> tuple[pin.Model, np.ndarray]:
        """Build a Pinocchio reduced model locking all non-actuated joints at neutral."""
        q_lock = pin.neutral(self.model)
        # joint ids to KEEP (actuated); all others get locked
        actuated_ids = [self.model.getJointId(n) for n in self.joint_names]
        joints_to_lock = [
            i for i in range(1, self.model.njoints)
            if i not in actuated_ids
        ]
        reduced_model = pin.buildReducedModel(
            self.model, joints_to_lock, q_lock
        )
        return reduced_model

    @property
    def reduced_model(self) -> pin.Model:
        """Pinocchio model with only the actuated joints (nq == len(joint_names))."""
        if not hasattr(self, '_reduced_model'):
            self._reduced_model = self._build_reduced_model()
        return self._reduced_model

    def reset(self):
        """Reset the robot to the start position."""
        self.forward_kinematics(self.start_position)

    def add_payload(self, frame_name: str, payload: pin.Inertia) -> None:
        """Add a payload inertia to the parent joint of the given frame."""
        if payload is None:
            raise ValueError(f"Payload is None. Check if the object exists in your Mujoco model.")
        frame_id = self.model.getFrameId(frame_name)
        if frame_id == -1:
            raise ValueError(f"Frame '{frame_name}' not found in the model.")
        parent_joint = self.model.frames[frame_id].parentJoint
        self.model.inertias[parent_joint] += payload
        self.data = self.model.createData()

    def forward_kinematics(self, q_reduced: np.ndarray) -> None:
        """Run FK with a reduced joint position vector (actuated joints only)."""
        self._q_reduced = np.asarray(q_reduced).copy()
        q = pin.neutral(self.model)
        q[self.q_indices] = self._q_reduced
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def current_q(self) -> np.ndarray:
        """Return the current reduced joint configuration (set by the last forward_kinematics call)."""
        return self._q_reduced.copy()

    def frame_position(self, frame_name: str) -> np.ndarray:
        """Return 3D world position of a frame (call forward_kinematics first)."""
        frame_id = self.model.getFrameId(frame_name)
        return self.data.oMf[frame_id].translation.copy()
