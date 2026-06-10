import mujoco.viewer
import time

class Visualizer:
    def __init__(self, model, data):
        self.model = model
        self.data = data

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

    def visualize(self, mode, vector, dt=0.01):
        '''
        Visualize the trajectory in MuJoCo viewer.
        Args:
            mode (str): "position" or "control" to specify the type of trajectory.
            vector (np.ndarray): The trajectory to visualize. For "position", it should be of shape (T, nq). For "control", it should be of shape (T, nu).
            dt (float): Time step between frames in seconds.
        '''
        if mode == "position" and vector is not None:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self._setup_camera(viewer)
                for i in range(len(vector)):
                    self.data.qpos[:] = vector[i][: self.model.nq]
                    mujoco.mj_forward(self.model, self.data)
                    viewer.sync()
                    time.sleep(dt)
                input("Press Enter to continue...")
                return vector
        elif mode == "control" and vector is not None:
            x_pos = []
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self._setup_camera(viewer)
                for i in range(len(vector)):
                    self.data.ctrl[:] = vector[i]
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    time.sleep(dt)
                    x_pos.append(self.data.qpos.copy())
                input("Press Enter to continue...")
                return x_pos
