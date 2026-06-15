import mujoco.viewer
import time
import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
import scipy.spatial.transform

class RobotMujocoModel:
    def __init__(self, path, list_of_joints=None, start_position=None):
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        if list_of_joints is None:
            list_of_joints = self._actuated_joint_names()
        assert type(list_of_joints) == list, "list_of_joints should be a list of joint names."
        self.joint_names = list_of_joints
        self.q_indices = self._q_indices(list_of_joints)
        self.v_indices = self._v_indices(list_of_joints)
        self.u_indices = self._u_indices(list_of_joints)
        self.start_position = start_position if start_position is not None else np.zeros(len(self.q_indices))
        self.set_joint_positions(self.start_position)

        self.lookat=np.array([0.0, 0.0, 0.7])
        self.distance=3.5
        self.azimuth=90
        self.elevation=-30

    def _actuated_joint_names(self) -> list[str]:
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.model.actuator_trnid[i, 0])
            for i in range(self.model.nu)
            if self.model.actuator_trnid[i, 0] != -1
        ]
    
    def _q_indices(self, list_of_joints=None) -> np.ndarray:
        if list_of_joints is None:
            list_of_joints = self.joint_names
        return np.array([self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)] for joint_name in list_of_joints], dtype=int)
    
    def _v_indices(self, list_of_joints=None) -> np.ndarray:
        if list_of_joints is None:
            list_of_joints = self.joint_names
        return np.array([self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)] for joint_name in list_of_joints], dtype=int)

    def _u_indices(self, list_of_joints=None) -> np.ndarray:
        if list_of_joints is None:
            list_of_joints = self.joint_names
        joint_to_actuator = {
            self.model.actuator_trnid[i, 0]: i
            for i in range(self.model.nu)
            if self.model.actuator_trnid[i, 0] != -1
        }
        return np.array([
            joint_to_actuator[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in list_of_joints
        ], dtype=int)

    def _setup_camera(self, viewer):
        viewer.cam.lookat[:] = self.lookat   # look-at point
        viewer.cam.distance = self.distance             # distance from lookat
        viewer.cam.azimuth = self.azimuth               # horizontal angle (degrees)
        viewer.cam.elevation = self.elevation            # vertical angle (degrees)

    def get_obj_payload(self, obj_name):
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if obj_id == -1:
            raise ValueError(f"Object '{obj_name}' not found in the model. Check if the name is correct and exists in your Mujoco model.")
        # Read inertial properties from MuJoCo
        mj_mass    = self.model.body_mass[obj_id]                   # scalar [kg]
        mj_com     = self.model.body_ipos[obj_id].copy()            # CoM in body frame [m]
        mj_diag_I  = self.model.body_inertia[obj_id].copy()        # principal moments [kg·m²]
        mj_iquat   = self.model.body_iquat[obj_id].copy()           # (w,x,y,z) quaternion of inertia frame

        w, x, y, z = mj_iquat
        R = scipy.spatial.transform.Rotation.from_quat([x, y, z, w]).as_matrix()

        I_full = R @ np.diag(mj_diag_I) @ R.T
        return pin.Inertia(mj_mass, mj_com, I_full)

    def reset(self) -> None:
        self.data.qpos[self.q_indices] = self.start_position
        self.data.qvel[self.v_indices] = 0.0
        self.data.ctrl[self.u_indices] = 0.0
        mujoco.mj_forward(self.model, self.data) 

    def current_state(self) -> np.ndarray:
        return np.concatenate([self.data.qpos[self.q_indices], self.data.qvel[self.v_indices]])

    def current_joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.q_indices]
    
    def current_joint_velocities(self) -> np.ndarray:
        return self.data.qvel[self.v_indices]

    def set_joint_positions(self, q: np.ndarray) -> None:
        if len(q) != len(self.q_indices):
            raise ValueError(f"Input q must have length {len(self.q_indices)}, but got {len(q)}.")
        self.data.qpos[self.q_indices] = q
        mujoco.mj_forward(self.model, self.data)

    def snapshot(self, plot: bool = False) -> np.ndarray:
        renderer = mujoco.Renderer(self.model, height=480, width=640)
        renderer.update_scene(self.data)
        plt.imshow(renderer.render())
        plt.axis('off')
        plt.show()

    def setup_camera(self, lookat=[0, 0, 0.7], distance=3.5, azimuth=180, elevation=-40):
        self.lookat = lookat
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation

    def visualize(self, mode, xs=None, us=None, dt=0.01, hold=True, kp=None, kd=None, plot=False):
        '''
        Visualize the trajectory in MuJoCo viewer.
        Args:
            mode (str): "position" or "control".
            xs: desired state trajectory for feedforward+feedback correction (optional,
                only used in "control" mode). Each element should be [q_des, v_des] of
                length >= 2*nu. When provided, a PD correction term is added to each u:
                  u = u_ff + kp*(q_des - q) + kd*(v_des - v)
            us: desired control trajectory for feedforward+feedback correction (optional,
                only used in "control" mode). Each element should be [u_ff] of
                length >= nu. When provided, a PD correction term is added to each u:
                  u = u_ff + kp*(q_des - q) + kd*(v_des - v)
            kp, kd: PD gains (np.ndarray of length nu). Required when xs is provided.
        '''
        x_real = []
        u_real = []
        if mode == "position" and xs is not None:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self._setup_camera(viewer)
                for i in range(len(xs)):
                    self.data.qpos[self.q_indices] = xs[i][:len(self.q_indices)]
                    mujoco.mj_forward(self.model, self.data)
                    viewer.sync()
                    time.sleep(dt)
                    x_real.append(np.concatenate([
                        self.data.qpos[self.q_indices],
                        self.data.qvel[self.v_indices],
                    ]))
                    u_real.append(self.data.ctrl[self.u_indices].copy())
                if hold:
                    input("Press Enter to continue...")
                return x_real, u_real
        elif mode == "control" and us is not None:
            use_feedback = xs is not None and kp is not None and kd is not None
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self._setup_camera(viewer)
                closed_loop_part = []
                for i in range(len(us)):
                    u_computed = np.array(us[i], dtype=float)
                    if use_feedback:  # Apply feedback every 2 steps to reduce computation
                        nq = len(self.q_indices)
                        x_des = np.asarray(xs[i])
                        q_des = x_des[:nq]
                        v_des = x_des[nq:2*nq]
                        q = self.data.qpos[self.q_indices]
                        v = self.data.qvel[self.v_indices]
                        correction = kp * (q_des - q) + kd * (v_des - v)
                        u = u_computed + correction
                        u = np.clip(u, self.model.actuator_ctrlrange[self.u_indices, 0], self.model.actuator_ctrlrange[self.u_indices, 1])
                        closed_loop_part.append(100 * np.abs(u - u_computed) / (np.abs(u_computed) + 1e-6))  # feedback contribution in percentage
                    else:
                        u = u_computed
                    self.data.ctrl[self.u_indices] = u
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    time.sleep(dt)
                    x_real.append(np.concatenate([
                        self.data.qpos[self.q_indices],
                        self.data.qvel[self.v_indices],
                    ]))
                    u_real.append(self.data.ctrl[self.u_indices].copy())
                if hold:
                    input("Press Enter to continue...")
                if plot and use_feedback:
                    closed_loop_part = np.array(closed_loop_part)
                    plt.figure(figsize=(10, 6))
                    for i in range(closed_loop_part.shape[1]):
                        plt.plot(closed_loop_part[:, i], label=f'Joint {i + 1}')
                    plt.xlabel('Time step')
                    plt.ylabel('Feedback contribution (%)')
                    plt.title('Feedback Contribution to Control')
                    plt.ylim(0, 100)
                    plt.legend()
                    plt.grid()
                    plt.show()
                return x_real, u_real
            
    def mpc_visualize(self, path, compute_control : callable, segment_lengths=None, dt=0.01, hold=True, step=1, zoh=True):
        """
        Args:
            step: number of sim steps between OCP re-solves.
            zoh:  True  (default) → Zero-Order Hold: apply u_computed[0] for ALL `step`
                          sim steps between re-solves.  Use when DT_ocp = step * sim_dt.
                          Prevents chattering from bang-bang OCP plan oscillations.
                  False → plan-following: apply u_computed[0], [1], … sequentially.
                          Only correct when DT_ocp == sim_dt.
        """
        if compute_control is None:
            raise ValueError("compute_control function must be provided for MPC mode.")
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self._setup_camera(viewer)
            x_real = []
            u_real = []
            warm_xs, warm_us = None, None
            x_computed = None
            u_computed = np.zeros((1, len(self.u_indices)))
            for i in range(len(path) - 1):
                for _ in range(segment_lengths[i]):
                    u_idx = _ % step
                    if u_idx == 0:
                        q_current = self.data.qpos[self.q_indices].copy()
                        v_current = self.data.qvel[self.v_indices].copy()
                        x_current = np.concatenate([q_current, v_current])
                        x_des = np.concatenate([np.asarray(path[i+1]), np.zeros(len(self.v_indices))])
                        # Shifted warm-start: drop first `step` nodes, pad tail, prepend x_current.
                        # warm_xs needs T+1 rows; warm_us needs T rows (Crocoddyl ShootingProblem).
                        if x_computed is not None and len(u_computed) >= step:
                            warm_xs = np.concatenate([
                                [x_current],
                                x_computed[step:],
                                np.tile(x_computed[-1:], (step, 1)),
                            ])
                            warm_us = np.concatenate([
                                u_computed[step:],
                                np.tile(u_computed[-1:], (step, 1)),
                            ])
                        else:
                            warm_xs, warm_us = None, None
                        x_computed, u_computed = compute_control(warm_xs, warm_us, x_current, x_des)
                        u_computed = np.asarray(u_computed, dtype=float)
                        u_computed = np.clip(u_computed, self.model.actuator_ctrlrange[self.u_indices, 0], self.model.actuator_ctrlrange[self.u_indices, 1])
                    if zoh:
                        # ZOH: hold u[0] for the whole interval.
                        # Prevents chattering: consecutive OCP controls u[k] can flip sign
                        # at small DT because the linearised dynamics alternate direction.
                        u_apply = u_computed[0]
                    else:
                        # Plan-following: only use when DT_ocp == sim_dt.
                        u_apply = u_computed[min(u_idx, len(u_computed) - 1)]
                    self.data.ctrl[self.u_indices] = u_apply
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    time.sleep(dt)
                    x_real.append(np.concatenate([
                        self.data.qpos[self.q_indices],
                        self.data.qvel[self.v_indices],
                    ]))
                    u_real.append(self.data.ctrl[self.u_indices].copy())
            if hold:
                input("Press Enter to continue...")
            return x_real, u_real
    

    def plot_controls(self, controls, dt=None):
        '''
        Plot the control trajectory using matplotlib.
        Args:
            controls (np.ndarray): The control trajectory of shape (T, nu).
        '''
        controls = np.asarray(controls)
        plt.figure(figsize=(10, 6))
        if dt is not None:
            time = np.arange(len(controls)) * dt
            plt.xlabel('Time (s)')
        else:
            time = np.arange(len(controls))
            plt.xlabel('Time step')
        for i in range(controls.shape[1]):
            plt.plot(time, controls[:, i], label=f'Joint {i + 1}')
        plt.ylabel('Control value')
        plt.title('Control Trajectory')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_error(self, x_pos, q_target, dt=None, segment_lengths=None):
        '''
        Plot configuration error over time.
        Args:
            x_pos: trajectory as returned by visualize() — rows are [q(nq), v(nq)].
            q_target: single target np.ndarray OR a list of targets (one per path segment).
            dt: timestep in seconds. If None, x-axis is step index.
            segment_lengths: list of int, number of steps per segment. Required when
                             q_target is a list. If omitted with a list, assumes equal segments.
        '''
        x_pos = np.array(x_pos)
        T = len(x_pos)

        if isinstance(q_target, (list, tuple)) and not isinstance(q_target[0], (int, float)):
            # multi-segment: build per-timestep target array
            targets = [np.asarray(q) for q in q_target]
            nq = len(targets[0])
            if segment_lengths is None:
                seg_len = T // len(targets)
                segment_lengths = [seg_len] * len(targets)
            q_target_per_step = np.concatenate([
                np.tile(t, (n, 1)) for t, n in zip(targets, segment_lengths)
            ])[:T]
        else:
            nq = len(q_target)
            q_target_per_step = np.tile(np.asarray(q_target), (T, 1))

        error = np.linalg.norm((x_pos[:, :nq] - q_target_per_step) * 180 / np.pi, axis=1)

        plt.figure(figsize=(10, 6))
        if dt is not None:
            t_axis = np.arange(T) * dt
            plt.plot(t_axis, error)
            plt.xlabel('Time (s)')
            # draw vertical lines at segment boundaries
            if isinstance(q_target, (list, tuple)) and not isinstance(q_target[0], (int, float)):
                cumulative = np.cumsum(segment_lengths[:-1]) * dt
                for t_sep in cumulative:
                    plt.axvline(x=t_sep, color='gray', linestyle='--', linewidth=0.8)
        else:
            plt.plot(error)
            plt.xlabel('Time step')
            if isinstance(q_target, (list, tuple)) and not isinstance(q_target[0], (int, float)):
                cumulative = np.cumsum(segment_lengths[:-1])
                for t_sep in cumulative:
                    plt.axvline(x=t_sep, color='gray', linestyle='--', linewidth=0.8)
        plt.ylabel('Configuration Error (degrees)')
        plt.title('Configuration Error Over Time')
        plt.grid()
        plt.show()