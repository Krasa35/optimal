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
        # SolverBoxFDDP reads u_lb/u_ub from the IntegratedActionModel, not from ActuationModel.
        u_lb = -self.robot_pin_model.reduced_model.effortLimit
        u_ub = self.robot_pin_model.reduced_model.effortLimit
        runningModel.u_lb = u_lb
        runningModel.u_ub = u_ub
        problem = crocoddyl.ShootingProblem(x_start, [runningModel] * T, terminalModel)
        solver = solver(problem)
        if warm_xs is not None and warm_us is not None:
            # Crocoddyl solve() expects list[np.ndarray], not a 2-D numpy array.
            warm_xs_list = [np.asarray(x, dtype=float) for x in warm_xs]
            warm_us_list = [np.clip(np.asarray(u, dtype=float), u_lb, u_ub) for u in warm_us]
            solver.solve(warm_xs_list, warm_us_list, 100, False)
        else:
            solver.solve()
        return np.array(solver.xs), np.array(solver.us)
    
    def _calc_crocoddylcontrol_advancedjoint(self, x_start, x_goal, T, DT, solver=crocoddyl.SolverBoxFDDP, warm_xs=None, warm_us=None, 
                                             track_weight=1e1, ctrl_weight=1e2, terminal_pose_weight=1e5, v_weight=1e2, acc_weight=1e3, terminal_v_weight=1e-1, **kwargs):
        # Use the reduced model (actuated joints only) so nq == len(joint_names)
        model = self.robot_pin_model.reduced_model
        nv = model.nv
        state = crocoddyl.StateMultibody(model)
        x_start = np.asarray(x_start, dtype=float)
        x_goal  = np.asarray(x_goal,  dtype=float)
        q_goal  = x_goal[:model.nq]

        # 1. Goal Tracking
        goalTrackingCost = crocoddyl.CostModelResidual(
            state, crocoddyl.ResidualModelState(state, x_goal))
        
        # 2. Gravity-Compensated Control Regularization
        try:
            uRegResidual = crocoddyl.ResidualModelControlGrav(state)
        except AttributeError:
            reduced_data = model.createData()
            u_ref = pin.computeGeneralizedGravity(model, reduced_data, q_goal)
            uRegResidual = crocoddyl.ResidualModelControl(state, u_ref)
        uRegCost = crocoddyl.CostModelResidual(state, uRegResidual)
        
        # 3. Velocity Regularization
        v_weights = np.concatenate([np.zeros(model.nq), np.ones(model.nv)])
        vActivation = crocoddyl.ActivationModelWeightedQuad(v_weights)
        vRegCost = crocoddyl.CostModelResidual(state, vActivation, crocoddyl.ResidualModelState(state, state.zero()))
        
        # 4. Joint Acceleration Regularization
        accResidual = crocoddyl.ResidualModelJointAcceleration(state)
        accRegCost = crocoddyl.CostModelResidual(state, accResidual)
        
        runningCostModel = crocoddyl.CostModelSum(state)
        terminalCostModel = crocoddyl.CostModelSum(state)

        runningCostModel.addCost("gripperPose", goalTrackingCost, track_weight)
        runningCostModel.addCost("ctrlReg", uRegCost, ctrl_weight)
        runningCostModel.addCost("vReg", vRegCost, weight=v_weight)
        runningCostModel.addCost("accReg", accRegCost, weight=acc_weight)
        
        terminalCostModel.addCost("gripperPose", goalTrackingCost, terminal_pose_weight)
        terminalCostModel.addCost("vReg", vRegCost, weight=terminal_v_weight)

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
        u_lb = -model.effortLimit
        u_ub =  model.effortLimit
        runningModel.u_lb = u_lb
        runningModel.u_ub = u_ub
        problem = crocoddyl.ShootingProblem(x_start, [runningModel] * T, terminalModel)
        _solver = solver(problem)
        if warm_xs is not None and warm_us is not None:
            warm_xs_list = [np.asarray(x, dtype=float) for x in warm_xs]
            warm_us_list = [np.clip(np.asarray(u, dtype=float), u_lb, u_ub) for u in warm_us]
            _solver.solve(warm_xs_list, warm_us_list, 100, False)
        else:
            _solver.solve()
        return np.array(_solver.xs), np.array(_solver.us)
    
    def update_pin_model(self, robot_pin_model):
        self.robot_pin_model = robot_pin_model
        self.robot_data = robot_pin_model.data
    
    def compute_control(self, x_start, x_goal, T, DT, option="pd", warm_xs=None, warm_us=None, **kwargs):
        self.robot_pin_model.forward_kinematics(x_start[:len(self.robot_pin_model.q_indices)])
        if option == "pd":
            xs, us = self._calc_pdcontrol(x_start, x_goal, T, DT, **kwargs)
            return xs[1:], us
        elif option == "crocoddyl":
            if warm_xs is None or warm_us is None:
                print("No warm start provided; computing warm start using PD control.")
                warm_xs, warm_us = self._calc_pdcontrol(x_start, x_goal, T, DT, **kwargs)
            xs, us = self._calc_crocoddylcontrol_basicjoint(x_start, x_goal, T, DT, warm_xs=warm_xs, warm_us=warm_us, **kwargs)
            return xs[1:], us
        elif option == "crocoddyl_advanced":
            if warm_xs is None or warm_us is None:
                print("No warm start provided; computing warm start using PD control.")
                warm_xs, warm_us = self._calc_pdcontrol(x_start, x_goal, T, DT, **kwargs)
            xs, us = self._calc_crocoddylcontrol_advancedjoint(x_start, x_goal, T, DT, warm_xs=warm_xs, warm_us=warm_us, **kwargs)
            return xs[1:], us
        else:
            raise ValueError(f"Unknown control option: {option}. Supported options are 'pd' and 'crocoddyl'.")
        