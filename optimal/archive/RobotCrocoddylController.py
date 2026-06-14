import crocoddyl
import numpy as np
from .RobotMujocoModel_old import RobotMujocoModel
from .RobotDiffModel import RobotDiffActionModel

class RobotCrocoddylController():
    def __init__(
        self,
        model: RobotMujocoModel,
        x_target: np.ndarray,
        dt: float | None = None,
        running_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        terminal_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        self.state = crocoddyl.StateVector(model.q_indices().size + model.v_indices().size)
        self.dt = model.model.opt.timestep if dt is None else float(dt)

        run_wq, run_wv, run_wu = running_weights
        term_wq, term_wv, term_wu = terminal_weights
        self.running_diff = RobotDiffActionModel(
            model, self.state, x_target, w_q=run_wq, w_v=run_wv, w_u=run_wu
        )
        self.terminal_diff = RobotDiffActionModel(
            model, self.state, x_target, w_q=term_wq, w_v=term_wv, w_u=term_wu
        )

        # RobotDiffActionModel.calcDiff is empty, so numerical derivatives are required.
        self.running_cost_model = crocoddyl.DifferentialActionModelNumDiff(self.running_diff, False)
        self.terminal_cost_model = crocoddyl.DifferentialActionModelNumDiff(self.terminal_diff, False)
        self.running_model = crocoddyl.IntegratedActionModelEuler(self.running_cost_model, self.dt)
        self.terminal_model = crocoddyl.IntegratedActionModelEuler(self.terminal_cost_model, 0.0)

    def set_target(self, x_target: np.ndarray) -> None:
        target = np.asarray(x_target, dtype=float).copy()
        self.running_diff.x_target = target
        self.terminal_diff.x_target = target


    def get_control(self, init_xs: list, init_us: list, x0: np.ndarray = None, use_previous_attrs: bool = False, problem: crocoddyl.ShootingProblem = None, solver: str = 'FDDP'):
        # Create the shooting problem
        if not use_previous_attrs:
            if problem is not None:
                self.problem = problem
            else:
                horizon = len(init_us) + 1
                if x0 is not None:
                    self.problem = crocoddyl.ShootingProblem(x0, [self.running_model] * (horizon - 1), self.terminal_model)
                else:
                    self.problem = crocoddyl.ShootingProblem(x0=np.zeros(self.state.nx), runningModels=[self.running_model] * (horizon - 1), terminalModel=self.terminal_model)
            # Create the solver
            if solver == 'FDDP':
                self.solver = crocoddyl.SolverFDDP(self.problem)
            elif solver == 'DDP':
                self.solver = crocoddyl.SolverDDP(self.problem)
            elif solver == 'BoxFDDP':
                self.solver = crocoddyl.SolverBoxFDDP(self.problem)
            else:
                raise ValueError("Solver not implemented")
        else:
            if not hasattr(self, "problem"):
                raise RuntimeError("No previous problem available. Call get_control with use_previous_attrs=False first.")
            self.problem.x0 = x0
        # Solve the problem
        # self.solver.setCallbacks([crocoddyl.CallbackVerbose()])
        self.solver.th_stop = 1e-9
        self.solver.solve(init_xs, init_us, 10, False)
        return self.solver.xs, self.solver.us
    

    def get_MPC_control(self, init_xs: list, init_us: list, x0: np.ndarray, problem: crocoddyl.ShootingProblem = None, solver: str = 'FDDP'):
        xs, us = self.get_control(init_xs, init_us, x0=x0, problem=problem, solver=solver)
        return xs[1:], us[0]