import os
import time
import numpy as np
import pinocchio as pin
import pybullet as p
import pybullet_data
import crocoddyl


class DifferentialFreeFwdDynamicsModelDerived(
    crocoddyl.DifferentialActionModelAbstract
):
    def __init__(self, state, actuationModel, costModel):
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, state, actuationModel.nu, costModel.nr
        )
        self.actuation = actuationModel
        self.costs = costModel
        self.enable_force = True
        self.armature = np.matrix(np.zeros(0))

    def calc(self, data, x, u=None):
        if u is None:
            q, v = x[: self.state.nq], x[-self.state.nv :]
            pin.computeAllTerms(self.state.pinocchio, data.pinocchio, q, v)
            self.costs.calc(data.costs, x)
            data.cost = data.costs.cost
        else:
            q, v = x[: self.state.nq], x[-self.state.nv :]
            self.actuation.calc(data.actuation, x, u)
            tau = data.actuation.tau
            # Computing the dynamics using ABA or manually for armature case
            if self.enable_force:
                data.xout[:] = pin.aba(
                    self.state.pinocchio, data.pinocchio, q, v, tau
                )
            else:
                pin.computeAllTerms(self.state.pinocchio, data.pinocchio, q, v)
                data.M = data.pinocchio.M
                if self.armature.size == self.state.nv:
                    data.M[range(self.state.nv), range(self.state.nv)] += self.armature
                data.Minv = np.linalg.inv(data.M)
                data.xout[:] = np.dot(data.Minv, (tau - data.pinocchio.nle))
            # Computing the cost value and residuals
            # Note: No Cartesian kinematics computations are needed when using
            # only configuration space (joint space) costs.
            self.costs.calc(data.costs, x, u)
            data.cost = data.costs.cost

    def calcDiff(self, data, x, u=None):
        if u is None:
            self.costs.calcDiff(data.costs, x)
        else:
            nq, nv = self.state.nq, self.state.nv
            q, v = x[:nq], x[-nv:]
            # Computing the actuation derivatives
            self.actuation.calcDiff(data.actuation, x, u)
            tau = data.actuation.tau
            # Computing the dynamics derivatives
            if self.enable_force:
                pin.computeABADerivatives(
                    self.state.pinocchio, data.pinocchio, q, v, tau
                )
                ddq_dq = data.pinocchio.ddq_dq
                ddq_dv = data.pinocchio.ddq_dv
                data.Fx[:, :] = np.hstack([ddq_dq, ddq_dv]) + np.dot(
                    data.pinocchio.Minv, data.actuation.dtau_dx
                )
                data.Fu[:, :] = np.dot(data.pinocchio.Minv, data.actuation.dtau_du)
            else:
                pin.computeRNEADerivatives(
                    self.state.pinocchio, data.pinocchio, q, v, data.xout
                )
                ddq_dq = np.dot(
                    data.Minv, (data.actuation.dtau_dx[:, :nv] - data.pinocchio.dtau_dq)
                )
                ddq_dv = np.dot(
                    data.Minv, (data.actuation.dtau_dx[:, nv:] - data.pinocchio.dtau_dv)
                )
                data.Fx[:, :] = np.hstack([ddq_dq, ddq_dv])
                data.Fu[:, :] = np.dot(data.Minv, data.actuation.dtau_du)
            # Computing the cost derivatives
            self.costs.calcDiff(data.costs, x, u)

    def createData(self):
        data = DifferentialFreeFwdDynamicsDataDerived(self)
        return data

    def set_armature(self, armature):
        if armature.size is not self.state.nv:
            print("The armature dimension is wrong, we cannot set it.")
        else:
            self.enable_force = False
            self.armature = armature.T


class DifferentialFreeFwdDynamicsDataDerived(crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        self.pinocchio = pin.Model.createData(model.state.pinocchio)
        self.multibody = crocoddyl.DataCollectorMultibody(self.pinocchio)
        self.actuation = model.actuation.createData()
        self.costs = model.costs.createData(self.multibody)
        self.costs.shareMemory(self)
        self.Minv = None


def main():
    print("--- SOLVING OPTIMAL CONTROL IN CONFIGURATION SPACE ---")
    
    urdf_path = "../models/ur10/ur10.urdf" 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_full_path = os.path.join(script_dir, urdf_path)
    
    # Load model and set gravity
    pin_model = pin.buildModelFromUrdf(urdf_full_path)
    pin_model.gravity = pin.Motion(np.array([0, 0, -9.81, 0, 0, 0]))
    
    # Setup state and actuation
    state = crocoddyl.StateMultibody(pin_model)
    actuation = crocoddyl.ActuationModelFull(state)
    nv, nu = state.nv, state.nv
    dt_plan = 0.01
    T = 100
    
    # Initial state
    q0 = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    x0 = np.concatenate([q0, np.zeros(nv)])
    
    # Target joint configuration
    q_target = np.array([1.0, -1.0, 1.0, -1.0, -1.0, 1.0])
    x_target = np.concatenate([q_target, np.zeros(nv)])
    
    # 1. State Tracking Cost (Target Configuration)
    # Using ResidualModelState to track the target joint positions and zero velocity
    stateResidual = crocoddyl.ResidualModelState(state, x_target, nu)
    stateCost = crocoddyl.CostModelResidual(state, stateResidual)
    
    # 2. Regularization Costs
    xRegResidual = crocoddyl.ResidualModelState(state, x0, nu)
    uRegResidual = crocoddyl.ResidualModelJointEffort(state, actuation, nu)
    
    xRegCost = crocoddyl.CostModelResidual(state, xRegResidual)
    uRegCost = crocoddyl.CostModelResidual(state, uRegResidual)
    
    # Sum models
    runningCosts = crocoddyl.CostModelSum(state)
    terminalCosts = crocoddyl.CostModelSum(state)
    
    # Running costs
    runningCosts.addCost("uReg", uRegCost, 1e-4)
    runningCosts.addCost("xReg", xRegCost, 1e-3)
    runningCosts.addCost("stateTracking", stateCost, 1e-1) # Optional: track target during flight
    
    # Terminal costs: heavy weight to ensure we reach the target configuration
    terminalCosts.addCost("goalState", stateCost, 1e4)
    
    # Action models using our configuration-space optimized dynamics class
    runningModel = crocoddyl.IntegratedActionModelEuler(
        DifferentialFreeFwdDynamicsModelDerived(state, actuation, runningCosts), dt_plan
    )
    terminalModel = crocoddyl.IntegratedActionModelEuler(
        DifferentialFreeFwdDynamicsModelDerived(state, actuation, terminalCosts), 0.0
    )
    
    # Setup problem and solver
    problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
    solver = crocoddyl.SolverFDDP(problem)
    solver.setCallbacks([crocoddyl.CallbackVerbose()])
    
    # Warm-start
    xs_init = [x0] * (T + 1)
    us_init = [np.zeros(nu)] * T
    
    print(f"Initial configuration: {np.round(q0, 2)}")
    print(f"Target configuration:  {np.round(q_target, 2)}")
    print("Solving trajectory optimization...")
    
    solver.solve(xs_init, us_init, maxiter=200)
    
    q_final = solver.xs[-1][:nv]
    print("\nOptimization complete!")
    print(f"Final reached config:  {np.round(q_final, 2)}")
    print(f"Target configuration:  {np.round(q_target, 2)}")
    print(f"Error (L2 norm):       {np.linalg.norm(q_final - q_target):.6f}")

if __name__ == '__main__':
    main()
