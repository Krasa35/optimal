import mujoco.viewer
import time
import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.lookat=np.array([0.0, 0.0, 0.7])
        self.distance=3.5
        self.azimuth=90
        self.elevation=-30

    def setup_camera(self, lookat=[0, 0, 0.7], distance=3.5, azimuth=180, elevation=-40):
        self.lookat = lookat
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation

    def _setup_camera(self, viewer):
        viewer.cam.lookat[:] = self.lookat   # look-at point
        viewer.cam.distance = self.distance             # distance from lookat
        viewer.cam.azimuth = self.azimuth               # horizontal angle (degrees)
        viewer.cam.elevation = self.elevation            # vertical angle (degrees)

    def visualize(self, mode, vector, dt=0.01, hold=True):
        '''
        Visualize the trajectory in MuJoCo viewer.
        Args:
            mode (str): "position" or "control" to specify the type of trajectory.
            vector (np.ndarray): The trajectory to visualize. For "position", it should be of shape (T, nq). For "control", it should be of shape (T, nu).
            dt (float): Time step between frames in seconds.
        '''
        x_pos = []
        if mode == "position" and vector is not None:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self._setup_camera(viewer)
                for i in range(len(vector)):
                    self.data.qpos[:] = vector[i][: self.model.nq]
                    mujoco.mj_forward(self.model, self.data)
                    viewer.sync()
                    time.sleep(dt)
                    x_pos.append(self.data.qpos.copy())
                if hold:
                    input("Press Enter to continue...")
                return x_pos
        elif mode == "control" and vector is not None:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self._setup_camera(viewer)
                for i in range(len(vector)):
                    self.data.ctrl[:] = vector[i]
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    time.sleep(dt)
                    x_pos.append(self.data.qpos.copy())
                if hold:
                    input("Press Enter to continue...")
                return x_pos

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

    def plot_error(self, x_pos, q_target, dt=None):
        x_pos = np.asarray(x_pos)
        q_arr = np.full_like(x_pos, q_target)  # Create an array of the same shape as x_pos filled with q_target
        error = np.linalg.norm((x_pos[:, :4] - q_arr[:, :4]) * 180 / np.pi, axis=1)
        plt.figure(figsize=(10, 6))
        if dt is not None:
            time = np.arange(len(error)) * dt
            plt.plot(time, error)
            plt.xlabel('Time (s)')
        else:
            plt.plot(error)
            plt.xlabel('Time step')
        plt.ylabel('Configuration Error (degrees)')
        plt.title('Configuration Error Over Time')
        plt.grid()
        plt.show()