import crocoddyl
import numpy as np
from RobotMujocoModel import RobotMujocoModel
from RobotDiffModel import RobotDiffActionModel

class RobotCrocoddylController():
    def __init__(self, model: RobotMujocoModel, x_target: np.ndarray):
        self.state = crocoddyl.StateVector(model.q_indices().size + model.v_indices().size)
        
        self.running_cost_model = RobotDiffActionModel(model, self.state, x_target)
        self.terminal_cost_model = RobotDiffActionModel(model, self.state, x_target)
        self.running_model = crocoddyl.IntegratedActionModelEuler(self.running_cost_model)
        self.terminal_model = crocoddyl.IntegratedActionModelEuler(self.terminal_cost_model)

        self.problem = crocoddyl.ShootingProblem(x0=np.zeros(self.state.nx), T=10, runningModels=[self.running_model] * 9, terminalModel=self.terminal_model)
        self.solver = crocoddyl.SolverFDDP(self.problem)
        self.solver.setCallbacks([crocoddyl.CallbackVerbose()])

        