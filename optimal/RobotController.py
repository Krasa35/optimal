from optimal.PDController import PDController
from optimal.RobotPinocchioModel import RobotPinocchioModel
import pinocchio as pin
import numpy as np
import crocoddyl


class RobotController:
    def __init__(self, robot_pin_model: RobotPinocchioModel):
        self.robot_pin_model = robot_pin_model
        self.robot_data = robot_pin_model.data

    def _calc_pdcontrol(self, q_start, q_target, T, DT, kp, kd, alpha=0.0, **kwargs):
        pd_controller = PDController(self.robot_pin_model.reduced_model, kp=kp, kd=kd)
        xs, us = pd_controller.compute_control(x_start=q_start, x_goal=q_target, horizon=T, control_dt=DT, alpha=alpha)
        return xs, us
    
    def _calc_crocoddylcontrol(self, q_start, q_target, T, DT, warm_xs=None, warm_us=None, track_weight=1e2, ctrl_weight=5e-2, terminal_weight=1e5, **kwargs):
        # Use the reduced model (actuated joints only) so nq == len(joint_names)
        model = self.robot_pin_model.reduced_model
        nv = model.nv
        state = crocoddyl.StateMultibody(model)
        framePlacementResidual = crocoddyl.ResidualModelState(state, np.concatenate([q_target, np.zeros(nv)]))
        goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
        uRegCost = crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelControl(state))
        
        runningCostModel = crocoddyl.CostModelSum(state)
        terminalCostModel = crocoddyl.CostModelSum(state)

        runningCostModel.addCost("gripperPose", goalTrackingCost, track_weight)
        runningCostModel.addCost("ctrlReg", uRegCost, ctrl_weight)
        terminalCostModel.addCost("gripperPose", goalTrackingCost, terminal_weight)

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
        x0 = np.concatenate([q_start, np.zeros(nv)])
        problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
        solver = crocoddyl.SolverBoxFDDP(problem)
        if warm_xs is not None and warm_us is not None:
            solver.solve(warm_xs, warm_us, 100, False)
        else:
            solver.solve()
        return solver.xs, solver.us
    
    def update_pin_model(self, robot_pin_model):
        self.robot_pin_model = robot_pin_model
        self.robot_data = robot_pin_model.data
    
    def compute_control(self, q_start, q_target, T, DT, option="pd", **kwargs):
        self.robot_pin_model.forward_kinematics(q_start)
        if option == "pd":
            xs, us = self._calc_pdcontrol(q_start, q_target, T, DT, **kwargs)
            return xs, us
        elif option == "boxfddp":
            warm_xs, warm_us = self._calc_pdcontrol(q_start, q_target, T, DT, **kwargs)
            xs, us = self._calc_crocoddylcontrol(q_start, q_target, T, DT, warm_xs=warm_xs, warm_us=warm_us, **kwargs)
            return xs, us
        