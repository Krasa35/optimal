from optimal.PDController import PDController
from optimal.RobotPinocchioModel import RobotPinocchioModel
import pinocchio as pin
import numpy as np
import crocoddyl


class RobotController:
    def __init__(self, robot_pin_model: RobotPinocchioModel):
        self.robot_pin_model = robot_pin_model
        self.robot_data = robot_pin_model.data

    def _calc_pdcontrol(self, x_start, x_goal, T, DT, kp, kd, alpha=0.0, **kwargs):
        pd_controller = PDController(self.robot_pin_model.reduced_model, kp=kp, kd=kd)
        xs, us = pd_controller.compute_control(x_start=x_start, x_goal=x_goal, horizon=T, control_dt=DT, alpha=alpha)
        return xs, us
    
    def _calc_crocoddylcontrol_basicjoint(self, x_start, x_goal, T, DT, solver=crocoddyl.SolverBoxFDDP, warm_xs=None, warm_us=None, track_weight=1e2, ctrl_weight=5e-2, terminal_weight=1e5, **kwargs):
        # Use the reduced model (actuated joints only) so nq == len(joint_names)
        model = self.robot_pin_model.reduced_model
        nv = model.nv
        state = crocoddyl.StateMultibody(model)
        framePlacementResidual = crocoddyl.ResidualModelState(state, x_goal)
        goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
        uRegCost = crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelControl(state))
        
        runningCostModel = crocoddyl.CostModelSum(state)
        terminalCostModel = crocoddyl.CostModelSum(state)

        runningCostModel.addCost("gripperPose", goalTrackingCost, track_weight)
        runningCostModel.addCost("ctrlReg", uRegCost, ctrl_weight)
        terminalCostModel.addCost("gripperPose", goalTrackingCost, terminal_weight)

        actuationModel = crocoddyl.ActuationModelFull(state)
        actuationModel.u_lb = -self.robot_pin_model.model.effortLimit[self.robot_pin_model.v_indices]
        actuationModel.u_ub = self.robot_pin_model.model.effortLimit[self.robot_pin_model.v_indices]
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
        problem = crocoddyl.ShootingProblem(x_start, [runningModel] * T, terminalModel)
        solver = solver(problem)
        if warm_xs is not None and warm_us is not None:
            solver.solve(warm_xs, warm_us, 100, False)
        else:
            solver.solve()
        return np.array(solver.xs), np.array(solver.us)
    
    def _calc_crocoddylcontrol_advancedjoint(self, q_start, q_target, T, DT, solver=crocoddyl.SolverBoxFDDP, warm_xs=None, warm_us=None, track_weight=1e2, ctrl_weight=5e-2, terminal_weight=1e5, **kwargs):
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
        _solver = solver(problem)
        if warm_xs is not None and warm_us is not None:
            _solver.solve(warm_xs, warm_us, 100, False)
        else:
            _solver.solve()
        return _solver.xs, _solver.us
    
    def update_pin_model(self, robot_pin_model):
        self.robot_pin_model = robot_pin_model
        self.robot_data = robot_pin_model.data
    
    def compute_control(self, x_start, x_goal, T, DT, option="pd", **kwargs):
        self.robot_pin_model.forward_kinematics(x_start[:len(self.robot_pin_model.q_indices)])
        if option == "pd":
            xs, us = self._calc_pdcontrol(x_start, x_goal, T, DT, **kwargs)
            return xs[1:], us
        elif option == "crocoddyl":
            warm_xs, warm_us = self._calc_pdcontrol(x_start, x_goal, T, DT, **kwargs)
            xs, us = self._calc_crocoddylcontrol_basicjoint(x_start, x_goal, T, DT, warm_xs=warm_xs, warm_us=warm_us, **kwargs)
            return xs[1:], us
        else:
            raise ValueError(f"Unknown control option: {option}. Supported options are 'pd' and 'crocoddyl'.")
        