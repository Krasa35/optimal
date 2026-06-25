import numpy as np

class UR:
    list_of_joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    start_position = np.array([0.0, -np.pi/2, np.pi/2, 0.0, 0.0, 0.0])
    kp = np.array([400.0, 800.0, 400.0, 100.0, 4.0, .1])
    kd = np.array([40.0,  80.0,  40.0,  10.0,  .4,  .01])
    closed_loop_kp = np.array([4.0, 8.0, 4.0, 1.0, 1.0, 1.0])
    closed_loop_kd = np.array([.4, .8, .4, .1, .4, .1])
    
    

class UR10_sandbox(UR):
    mjcf_path = "../models/ur10/ur10_sandbox.xml"
    urdf_path = "../models/ur10/ur10_sandbox.urdf"

class UR10(UR):
    mjcf_path = "../models/ur10/ur10.xml"
    urdf_path = "../models/ur10/ur10.urdf"

class UR10_weighted(UR):
    mjcf_path = "../models/ur10/ur10_weighted.xml"
    urdf_path = "../models/ur10/ur10_sandbox.urdf"

class Trajectories:
    joint_6dof = [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([-1.8, -1.8, 1.8, -1.8, -1.8, 1.8]),
            np.array([-1.8, -1.2, 1.5, -1.8, -1.8, 1.8]),
            np.array([-1.8, -1.8, 1.8, -1.8, -1.8, 1.8]),
            np.array([0.4, -0.4, 1.5, -0.8, 1.5, 0.5])]
    
    joint_weighted = [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([-1.8, -1.8, 1.8, -1.8, -1.8, 1.8]),
            np.array([0.0, -1.8, 1.8, -1.8, -1.8, 1.8]),
            np.array([-1.8, -1.8, 1.8, -1.8, -1.8, 1.8]),
            np.array([-1.8, -1.2, 1.5, -1.8, -1.8, 1.8]),
            np.array([-1.8, -1.8, 1.8, -1.8, -1.8, 1.8]),
            np.array([0.4, -0.4, 1.5, -0.8, 1.5, 0.5])]


class Test1:
    class PDValues:
        class Stochastic:
            kp = np.array([400.0, 800.0, 400.0, 100.0, 4.0, 0.1])
            kd = np.array([40.0, 80.0, 40.0, 10.0, 0.4, 0.01])

        class GridSearch:
            class Linear:
                alpha = 1.0
                interpolation = "linear"
                kp = np.array([400.0, 400.0, 400.0, 100.0, 5.0, 5.0])
                kd = np.array([20.0, 40.0, 20.0, 5.0, 0.5, 0.25])

            class Cubic:
                alpha = 0.8
                interpolation = "cubic"
                kp = np.array([400.0, 400.0, 400.0, 100.0, 15.0, 5.0])
                kd = np.array([60.0, 60.0, 60.0, 15.0, 2.25, 0.25])

            class Trapezoidal:
                alpha = 0.8
                interpolation = "trapezoidal"
                kp = np.array([400.0, 400.0, 400.0, 100.0, 15.0, 5.0])
                kd = np.array([60.0, 60.0, 60.0, 15.0, 2.25, 0.25])

            class Quintic:
                alpha = 0.8
                interpolation = "quintic"
                kp = np.array([400.0, 400.0, 400.0, 100.0, 100.0, 5.0])
                kd = np.array([60.0, 60.0, 40.0, 10.0, 5.0, 0.5])

        class BayesianOptimization:

            class TPE:
                class Global:
                    alpha = 0.935
                    interpolation = "quintic"
                    kp = np.array([1414.94, 1693.23, 1202.19, 7.92, 34.51, 1.24])
                    kd = np.array([204.882, 113.588, 148.053, 0.687, 3.741, 0.178])

                class Piecewise:
                    class Segment1:
                        alpha = 0.591
                        interpolation = "quintic"
                        kp = np.array([1247.07, 1200.43, 759.28, 9.96, 9.21, 1.64])
                        kd = np.array([104.034, 72.139, 49.476, 0.844, 0.893, 0.157])

                    class Segment2:
                        alpha = 0.994
                        interpolation = "linear"
                        kp = np.array([307.69, 375.79, 24.17, 30.21, 13.17, 2.26])
                        kd = np.array([19.897, 21.817, 2.196, 1.929, 1.221, 0.174])

                    class Segment3:
                        alpha = 0.401
                        interpolation = "quintic"
                        kp = np.array([340.33, 266.66, 20.66, 13.25, 22.91, 2.37])
                        kd = np.array([46.229, 34.400, 1.383, 1.732, 1.651, 0.146])

                    class Segment4:
                        alpha = 0.771
                        interpolation = "trapezoidal"
                        kp = np.array([1360.92, 221.02, 556.68, 7.94, 18.93, 2.07])
                        kd = np.array([108.632, 21.676, 32.952, 1.033, 1.468, 0.300])

            class CMAES:
                class Global:
                    alpha = 0.681
                    interpolation = "cubic"
                    kp = np.array([743.38, 916.44, 1550.75, 48.37, 18.66, 3.50])
                    kd = np.array([73.175, 78.349, 192.953, 3.333, 1.308, 0.496])

                class Piecewise:
                    class Segment1:
                        alpha = 0.487
                        interpolation = "cubic"
                        kp = np.array([248.27, 16.68, 36.12, 5.21, 3.71, 1.91])
                        kd = np.array([24.187, 2.302, 2.753, 0.624, 0.321, 0.200])

                    class Segment2:
                        alpha = 0.732
                        interpolation = "cubic"
                        kp = np.array([178.97, 147.64, 1581.92, 9.75, 5.11, 2.76])
                        kd = np.array([13.011, 15.859, 161.079, 1.166, 0.511, 0.268])

                    class Segment3:
                        alpha = 0.606
                        interpolation = "trapezoidal"
                        kp = np.array([186.11, 1043.85, 355.85, 17.60, 5.01, 2.16])
                        kd = np.array([12.736, 103.732, 31.768, 1.909, 0.489, 0.158])

                    class Segment4:
                        alpha = 0.708
                        interpolation = "cubic"
                        kp = np.array([680.10, 666.82, 77.89, 4.53, 12.36, 3.30])
                        kd = np.array([58.363, 52.806, 10.477, 0.583, 1.022, 0.320])


    class CrocoddylValues:

        class GridSearch:

            class WithoutInterpolation:
                track_weight = 5.0e3
                ctrl_weight = 1.0e3
                terminal_pose_weight = 3.0e7
                v_weight = 5.0e-2
                acc_weight = 3.0e1

                search_time_s = 291.3
                best_score = 3.8417e4
                evaluations = 79

            class WithInterpolation:
                track_weight = 3.0e4
                ctrl_weight = 1.0e3
                terminal_pose_weight = 3.0e7
                v_weight = 1.0e-2
                acc_weight = 5.0

                search_time_s = 268.0
                best_score = 3.1761e4
                evaluations = 79

        class BayesianOptimization:

            class CMAES:
                class TrackInterp:

                    class FDDP:
                        track_w = 3990.0
                        ctrl_w = 1080.0
                        term_w = 21000000.0
                        v_w = 0.336
                        acc_w = 54.6

                    class BoxFDDP:
                        track_w = 3990.0
                        ctrl_w = 1080.0
                        term_w = 21000000.0
                        v_w = 0.336
                        acc_w = 54.6

                    class DDP:
                        track_w = 3670.0
                        ctrl_w = 418.0
                        term_w = 4740000.0
                        v_w = 0.125
                        acc_w = 6.44

                class NoTrackInterp:

                    class FDDP:
                        track_w = 2270.0
                        ctrl_w = 905.0
                        term_w = 23000000.0
                        v_w = 0.0510
                        acc_w = 65.4

                    class BoxFDDP:
                        track_w = 2270.0
                        ctrl_w = 905.0
                        term_w = 23000000.0
                        v_w = 0.0510
                        acc_w = 65.4

                    class DDP:
                        track_w = 12000.0
                        ctrl_w = 3160.0
                        term_w = 8170000.0
                        v_w = 0.659
                        acc_w = 4.55

            class TPE:
                class TrackInterp:

                    class BoxFDDP:
                        track_w = 5210.0
                        ctrl_w = 1070.0
                        term_w = 14100000.0
                        v_w = 0.0366
                        acc_w = 21.2

                    class FDDP:
                        track_w = 5210.0
                        ctrl_w = 1070.0
                        term_w = 14100000.0
                        v_w = 0.0366
                        acc_w = 21.2

                    class DDP:
                        track_w = 14300.0
                        ctrl_w = 1240.0
                        term_w = 13100000.0
                        v_w = 0.0171
                        acc_w = 1.72

                class NoTrackInterp:

                    class FDDP:
                        track_w = 1990.0
                        ctrl_w = 616.0
                        term_w = 90700000.0
                        v_w = 0.0220
                        acc_w = 37.2

                    class BoxFDDP:
                        track_w = 1990.0
                        ctrl_w = 616.0
                        term_w = 90700000.0
                        v_w = 0.0220
                        acc_w = 37.2

                    class DDP:
                        track_w = 19300.0
                        ctrl_w = 5310.0
                        term_w = 44100000.0
                        v_w = 0.626
                        acc_w = 7.08

class Test2:
    class WithPayload:
        class CMAES:
            class TrackInterp:
                class BoxFDDP:
                    track_w = 1080.0
                    ctrl_w = 2130.0
                    term_w = 35900000.0
                    v_w = 0.0336
                    acc_w = 6.13
                    term_v_w = 0.0607

        class TPE:
            class TrackInterp:
                class BoxFDDP:
                    track_w = 1070.0
                    ctrl_w = 2230.0
                    term_w = 61300000.0
                    v_w = 0.384
                    acc_w = 1.03
                    term_v_w = 0.979
    class WithoutPayload:
        class CMAES:
            class TrackInterp:
                class BoxFDDP:
                    track_w = 1080.0
                    ctrl_w = 2130.0
                    term_w = 35900000.0
                    v_w = 0.0336
                    acc_w = 6.13
                    term_v_w = 0.0607

        class TPE:
            class TrackInterp:
                class BoxFDDP:
                    track_w = 1070.0
                    ctrl_w = 2230.0
                    term_w = 61300000.0
                    v_w = 0.384
                    acc_w = 1.03
                    term_v_w = 0.979