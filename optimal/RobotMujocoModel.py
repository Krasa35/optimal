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

    def _simulate_control(self, us, xs=None, kp=None, kd=None, use_feedback=False):
        u_computed = np.array(us, dtype=float)
        if use_feedback:  # Apply feedback every 2 steps to reduce computation
            nq = len(self.q_indices)
            x_des = np.array(xs, dtype=float) 
            q_des = x_des[:nq]
            v_des = x_des[nq:2*nq]
            q = self.data.qpos[self.q_indices]
            v = self.data.qvel[self.v_indices]
            correction = kp * (q_des - q) + kd * (v_des - v)
            u = u_computed + correction
            u = np.clip(u, self.model.actuator_ctrlrange[self.u_indices, 0], self.model.actuator_ctrlrange[self.u_indices, 1])
        else:
            u = u_computed
        self.data.ctrl[self.u_indices] = u
        mujoco.mj_step(self.model, self.data)
        x_real = np.concatenate([self.data.qpos[self.q_indices], self.data.qvel[self.v_indices]])
        return x_real, u
    
    def simulate_control(self, us, xs=None, kp=None, kd=None):
        x_real, u_real = [], []
        use_feedback = xs is not None and kp is not None and kd is not None
        for i in range(len(us)):
            x, u = self._simulate_control(us[i], xs=xs[i] if xs is not None else None, kp=kp, kd=kd, use_feedback=use_feedback)
            x_real.append(x)
            u_real.append(u)
        return np.array(x_real), np.array(u_real)
    
    def plot_feedback_part(self, u_real, us, dt=None):
        u_tot = np.array(u_real, dtype=float)
        u_ff  = np.array(us,     dtype=float)
        u_fb  = u_tot - u_ff
        n_joints = u_tot.shape[1]
        time_axis = np.arange(len(u_tot)) * dt if dt is not None else np.arange(len(u_tot))

        # --- per-joint stats ---
        rms_tot = np.sqrt(np.mean(u_tot ** 2, axis=0))
        rms_ff  = np.sqrt(np.mean(u_ff  ** 2, axis=0))
        rms_fb  = np.sqrt(np.mean(u_fb  ** 2, axis=0))
        mae_fb  = np.mean(np.abs(u_fb),        axis=0)

        corr = np.array([
            np.corrcoef(u_ff[:, j], u_tot[:, j])[0, 1]
            for j in range(n_joints)
        ])

        print("=" * 55)
        print(f"{'Joint':<8} {'Corr FF↔tot':>12} {'RMS FF %':>9} {'RMS FB %':>9} {'MAE FB':>9}")
        print("-" * 55)
        for j in range(n_joints):
            denom = rms_tot[j] if rms_tot[j] > 0 else 1.0
            print(f"  {j+1:<6} {corr[j]:>12.4f} {rms_ff[j]/denom*100:>8.1f}% {rms_fb[j]/denom*100:>8.1f}% {mae_fb[j]:>9.4f}")
        print("=" * 55)
        g_rms_tot = np.linalg.norm(u_tot)
        g_rms_ff  = np.linalg.norm(u_ff)
        g_rms_fb  = np.linalg.norm(u_fb)
        print(f"Global FF coverage: {g_rms_ff/g_rms_tot*100:.1f}%   FB correction: {g_rms_fb/g_rms_tot*100:.1f}%")
        print(f"Global correlation FF↔tot: {np.corrcoef(u_ff.ravel(), u_tot.ravel())[0,1]:.4f}")

        # --- plot: FF vs total overlaid, correction below ---
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for j in range(n_joints):
            axes[0].plot(time_axis, u_tot[:, j], label=f'J{j+1} total', alpha=0.8)
        for j in range(n_joints):
            axes[0].plot(time_axis, u_ff[:, j],  label=f'J{j+1} FF',    linestyle='--', alpha=0.6)
        axes[0].set_ylabel('Torque [Nm]')
        axes[0].set_title('Feedforward vs Total control')
        axes[0].legend(ncol=n_joints, fontsize=8)
        axes[0].grid()

        for j in range(n_joints):
            axes[1].plot(time_axis, u_fb[:, j], label=f'J{j+1}  r={corr[j]:.3f}')
        axes[1].set_ylabel('FB correction [Nm]')
        if dt is not None:
            axes[1].set_xlabel('Time (s)')
        else:
            axes[1].set_xlabel('Time step')
        axes[1].set_title('Closed-loop correction  (u_total − u_ff)')
        axes[1].legend(ncol=n_joints, fontsize=8)
        axes[1].axhline(0, color='k', linewidth=0.5)
        axes[1].grid()

        # Add statistics text box
        stats_text = f"Global FF coverage: {g_rms_ff/g_rms_tot*100:.1f}%\n"
        stats_text += f"FB correction: {g_rms_fb/g_rms_tot*100:.1f}%\n"
        stats_text += f"Global correlation FF↔tot: {np.corrcoef(u_ff.ravel(), u_tot.ravel())[0,1]:.4f}"
        fig.text(0.99, 0.01, stats_text, ha='right', va='bottom', fontsize=9, 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), family='monospace')

        plt.tight_layout()
        plt.show()

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
                    self.set_joint_positions(xs[i][:len(self.q_indices)])
                    viewer.sync()
                    time.sleep(dt)
                    x_real.append(xs)
                    u_real.append(np.zeros(len(self.u_indices)))
        elif mode == "control" and us is not None:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                use_feedback = xs is not None and kp is not None and kd is not None
                self._setup_camera(viewer)
                for i in range(len(us)):
                    x, u = self._simulate_control(us[i], xs=xs[i] if xs is not None else None, kp=kp, kd=kd, use_feedback=use_feedback)
                    viewer.sync()
                    time.sleep(dt)
                    x_real.append(x)
                    u_real.append(u)
                if plot:
                    self.plot_feedback_part(u_real, us)
        if hold:
            input("Press Enter to continue...")
        return x_real, u_real
            
    def mpc_visualize(self, path, compute_control : callable, segment_lengths=None, dt=0.01, hold=True, step=1, zoh=True, speed=1.0):
        """
        Args:
            step: number of sim steps between OCP re-solves.
            zoh:  True  (default) → Zero-Order Hold: apply u_computed[0] for ALL `step`
                          sim steps between re-solves.  Use when DT_ocp = step * sim_dt.
                          Prevents chattering from bang-bang OCP plan oscillations.
                  False → plan-following: apply u_computed[0], [1], … sequentially.
                          Only correct when DT_ocp == sim_dt.
            speed: real-time factor for visualization wall-clock pacing.
                          1.0 → real time, 2.0 → 2× faster, 0.5 → slow-motion.
                          speed <= 0 → run as fast as possible (no pacing).
                          If a solve overruns its budget the loop falls behind and
                          stops sleeping until it catches back up (never sleeps negative).
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
            # Wall-clock pacing reference: wall time should track sim_time / speed.
            wall_start = time.perf_counter()
            sim_time = 0.0
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
                    # Real-time pacing: sleep only the time left until sim_time should be
                    # reached in wall-clock, scaled by `speed`. Never sleeps negative, so
                    # an overrunning solve simply makes the loop catch up at full speed.
                    sim_time += dt
                    if speed > 0:
                        target_wall = wall_start + sim_time / speed
                        remaining = target_wall - time.perf_counter()
                        if remaining > 0:
                            time.sleep(remaining)
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

    def plot_position(self, x_pos, dt=None):
        '''
        Plot the joint position trajectory using matplotlib.
        Args:
            x_pos (np.ndarray): The state trajectory of shape (T, 2*nq) where the first nq columns are joint positions.
            dt: timestep in seconds. If None, x-axis is step index.
        '''
        x_pos = np.asarray(x_pos)
        nq = len(self.q_indices)
        plt.figure(figsize=(10, 6))
        if dt is not None:
            time = np.arange(len(x_pos)) * dt
            plt.xlabel('Time (s)')
        else:
            time = np.arange(len(x_pos))
            plt.xlabel('Time step')
        for i in range(nq):
            plt.plot(time, x_pos[:, i], label=f'Joint {i + 1}')
        plt.ylabel('Joint Position (radians)')
        plt.title('Joint Position Trajectory')
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