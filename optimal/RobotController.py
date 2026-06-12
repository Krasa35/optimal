from optimal.PDController import PDController
import pinocchio as pin
import numpy as np
import crocoddyl


class RobotController:
    def __init__(self, robot_pin_model):
        self.robot_pin_model = robot_pin_model
        self.robot_data = pin.Data(robot_pin_model)

    def _calc_pdcontrol(self, q_start, q_target, T, DT,
                        kp=np.array([400.0, 800.0, 400.0, 100.0, 4.0, 1.0]),
                        kd=np.array([40.0,  80.0,  40.0,  10.0,  .4,  .1])):
        pd_controller = PDController(self.robot_pin_model, kp=kp, kd=kd)
        xs, us = pd_controller.compute_control(x_start=q_start, x_goal=q_target, horizon=T, control_dt=DT)
        return xs, us
    
    def compute_control(self, q_start, q_target, T, DT, option="pd", **kwargs):
        pin.forwardKinematics(self.robot_pin_model, self.robot_data, q_start)
        pin.updateFramePlacements(self.robot_pin_model, self.robot_data)
        xs, us = self._calc_pdcontrol(q_start, q_target, T, DT, **kwargs)
        if option == "pd":
            return xs, us
        elif option == "boxfddp":
            state = crocoddyl.StateMultibody(self.robot_pin_model)
            framePlacementResidual = crocoddyl.ResidualModelState(state, np.concatenate([q_target, np.zeros(self.robot_pin_model.nv)]))
            goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
            uRegCost = crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelControl(state))
            
            runningCostModel = crocoddyl.CostModelSum(state)
            terminalCostModel = crocoddyl.CostModelSum(state)

            runningCostModel.addCost("gripperPose", goalTrackingCost, 1e2)
            runningCostModel.addCost("ctrlReg", uRegCost, 5e-2)
            terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e5)

            actuationModel = crocoddyl.ActuationModelFull(state)
            runningModel = crocoddyl.IntegratedActionModelEuler(
                crocoddyl.DifferentialActionModelFreeFwdDynamics(
                    state, actuationModel, runningCostModel
                ),
                DT,
            )
            terminalModel = crocoddyl.IntegratedActionModelEuler(
                crocoddyl.DifferentialActionModelFreeFwdDynamics(
                    state, actuationModel, terminalCostModel
                ),
                0.0,
            )
            problem = crocoddyl.ShootingProblem(np.concatenate([q_start, np.zeros(self.robot_pin_model.nv)]), [runningModel] * T, terminalModel)
            solver = crocoddyl.SolverBoxFDDP(problem)
            solver.solve(xs, us, 100, False)
            return solver.xs, solver.us
        