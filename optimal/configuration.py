import numpy as np

class UR:
    list_of_joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    start_position = np.array([0.0, -np.pi/2, np.pi/2, 0.0, 0.0, 0.0])
    kp = np.array([400.0, 800.0, 400.0, 100.0, 4.0, 1.0])
    kd = np.array([40.0,  80.0,  40.0,  10.0,  .4,  .1])
    closed_loop_kp = np.array([4.0, 8.0, 4.0, 1.0, 1.0, 1.0])
    closed_loop_kd = np.array([.4, .8, .4, .1, .4, .1])

class UR10_sandbox(UR):
    mjcf_path = "../models/ur10/ur10_sandbox.xml"
    urdf_path = "../models/ur10/ur10_sandbox.urdf"

class UR10(UR):
    mjcf_path = "../models/ur10/ur10.xml"
    urdf_path = "../models/ur10/ur10.urdf"