from optimal.PDController import PDController
from optimal.RobotPinocchioModel import RobotPinocchioModel
import pinocchio as pin
import numpy as np
import crocoddyl


class RobotController:
    def __init__(self, robot_pin_model: RobotPinocchioModel):
        self.robot_pin_model = robot_pin_model
        self.robot_data = robot_pin_model.data

    def _calc_pdcontrol(self, x_start, x_goal, T, DT, kp, kd, alpha=0.0, interpolation="quintic", interpolation_method=None, **kwargs):
        pd_controller = PDController(self.robot_pin_model.reduced_model, kp=kp, kd=kd)
        xs, us = pd_controller.compute_control(x_start=x_start, x_goal=x_goal, horizon=T, control_dt=DT, alpha=alpha, interpolation=interpolation_method or interpolation)
        return xs, us
    
    def _calc_crocoddylcontrol_basicjoint(self, x_start, x_goal, T, DT, solver=crocoddyl.SolverBoxFDDP, warm_xs=None, warm_us=None, track_weight=1e2, ctrl_weight=5e-2, terminal_weight=1e5, maxiter=100, **kwargs):
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
            solver.solve(warm_xs_list, warm_us_list, maxiter, False)
        else:
            solver.solve()
        return np.array(solver.xs), np.array(solver.us)
    
    def _calc_crocoddylcontrol_advancedjoint(self, x_start, x_goal, T, DT, solver=crocoddyl.SolverBoxFDDP, warm_xs=None, warm_us=None,
                                             track_weight=1e3, ctrl_weight=1e1, terminal_pose_weight=1e6, v_weight=1e1, acc_weight=1e2, terminal_v_weight=1e1,
                                             track_interpolated=False, maxiter=100, **kwargs):
        # Use the reduced model (actuated joints only) so nq == len(joint_names)
        model = self.robot_pin_model.reduced_model
        nv = model.nv
        state = crocoddyl.StateMultibody(model)
        x_start = np.asarray(x_start, dtype=float)
        x_goal  = np.asarray(x_goal,  dtype=float)

        # Body / wrist split activation weights (position only, no velocity)
        body_weights  = np.array([1.0]*min(3, model.nq) + [0.0]*max(0, model.nq - 3) + [0.0]*model.nv)
        wrist_weights = np.array([0.0]*min(3, model.nq) + [1.0]*max(0, model.nq - 3) + [0.0]*model.nv)

        # 2. Gravity-Compensated Control Regularization
        try:
            uRegResidual = crocoddyl.ResidualModelControlGrav(state)
        except AttributeError:
            reduced_data = model.createData()
            u_ref = pin.computeGeneralizedGravity(model, reduced_data, x_goal)
            uRegResidual = crocoddyl.ResidualModelControl(state, u_ref)
        uRegCost = crocoddyl.CostModelResidual(state, uRegResidual)

        # 3. Velocity Regularization
        v_weights  = np.concatenate([np.zeros(model.nq), np.ones(model.nv)])
        vActivation = crocoddyl.ActivationModelWeightedQuad(v_weights)
        vRegCost = crocoddyl.CostModelResidual(state, vActivation, crocoddyl.ResidualModelState(state, state.zero()))

        # 4. Joint Acceleration Regularization
        accResidual = crocoddyl.ResidualModelJointAcceleration(state)
        accRegCost  = crocoddyl.CostModelResidual(state, accResidual)

        u_lb = -model.effortLimit
        u_ub =  model.effortLimit
        actuationModel = crocoddyl.ActuationModelFull(state)

        def _make_running_model(x_ref):
            stateRes  = crocoddyl.ResidualModelState(state, np.asarray(x_ref, dtype=float))
            cost_body  = crocoddyl.CostModelResidual(state, crocoddyl.ActivationModelWeightedQuad(body_weights),  stateRes)
            cost_wrist = crocoddyl.CostModelResidual(state, crocoddyl.ActivationModelWeightedQuad(wrist_weights), stateRes)
            cm = crocoddyl.CostModelSum(state)
            cm.addCost("gripperPose", cost_body,  track_weight)
            cm.addCost("wristPose",   cost_wrist, track_weight)
            cm.addCost("ctrlReg",     uRegCost,   ctrl_weight)
            cm.addCost("vReg",        vRegCost,   v_weight)
            cm.addCost("accReg",      accRegCost, acc_weight)
            rm = crocoddyl.IntegratedActionModelEuler(
                crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, cm), DT)
            rm.u_lb = u_lb
            rm.u_ub = u_ub
            return rm

        # 1. Running models — either time-varying (track_interpolated=True) or fixed x_goal
        if track_interpolated and warm_xs is not None and len(warm_xs) >= T:
            # Use the PD-interpolated state at each step as the per-timestep reference.
            # warm_xs[0] = x_start; running model t transitions from x[t] → x[t+1],
            # so the per-step reference is warm_xs[t+1] (desired arrival state).
            running_models = [_make_running_model(warm_xs[t + 1]) for t in range(T)]
        else:
            running_models = [_make_running_model(x_goal)] * T

        # Terminal model — always track x_goal
        stateRes_term   = crocoddyl.ResidualModelState(state, x_goal)
        cost_body_term  = crocoddyl.CostModelResidual(state, crocoddyl.ActivationModelWeightedQuad(body_weights),  stateRes_term)
        cost_wrist_term = crocoddyl.CostModelResidual(state, crocoddyl.ActivationModelWeightedQuad(wrist_weights), stateRes_term)
        terminalCostModel = crocoddyl.CostModelSum(state)
        terminalCostModel.addCost("gripperPose", cost_body_term,  terminal_pose_weight)
        terminalCostModel.addCost("wristPose",   cost_wrist_term, terminal_pose_weight)
        terminalCostModel.addCost("vReg",        vRegCost,        terminal_v_weight)
        terminalModel = crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, terminalCostModel), 0.0)

        problem = crocoddyl.ShootingProblem(x_start, running_models, terminalModel)
        _solver = solver(problem)
        if warm_xs is not None and warm_us is not None:
            warm_xs_list = [np.asarray(x, dtype=float) for x in warm_xs]
            warm_us_list = [np.clip(np.asarray(u, dtype=float), u_lb, u_ub) for u in warm_us]
            _solver.solve(warm_xs_list, warm_us_list, maxiter, False)
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
                warm_xs, warm_us = self._calc_pdcontrol(x_start, x_goal, T, DT, **kwargs)
            xs, us = self._calc_crocoddylcontrol_basicjoint(x_start, x_goal, T, DT, warm_xs=warm_xs, warm_us=warm_us, **kwargs)
            return xs[1:], us
        elif option == "crocoddyl_advanced":
            if warm_xs is None or warm_us is None:
                warm_xs, warm_us = self._calc_pdcontrol(x_start, x_goal, T, DT, **kwargs)
            xs, us = self._calc_crocoddylcontrol_advancedjoint(x_start, x_goal, T, DT, warm_xs=warm_xs, warm_us=warm_us, **kwargs)
            return xs[1:], us
        else:
            raise ValueError(f"Unknown control option: {option}. Supported options are 'pd', 'crocoddyl', and 'crocoddyl_advanced'.")
        