import crocoddyl
import pinocchio
import numpy as np

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
            pinocchio.computeAllTerms(self.state.pinocchio, data.pinocchio, q, v)
            self.costs.calc(data.costs, x)
            data.cost = data.costs.cost
        else:
            q, v = x[: self.state.nq], x[-self.state.nv :]
            self.actuation.calc(data.actuation, x, u)
            tau = data.actuation.tau
            # Computing the dynamics using ABA or manually for armature case
            if self.enable_force:
                data.xout[:] = pinocchio.aba(
                    self.state.pinocchio, data.pinocchio, q, v, tau
                )
            else:
                pinocchio.computeAllTerms(self.state.pinocchio, data.pinocchio, q, v)
                data.M = data.pinocchio.M
                if self.armature.size == self.state.nv:
                    data.M[range(self.state.nv), range(self.state.nv)] += self.armature
                data.Minv = np.linalg.inv(data.M)
                data.xout[:] = np.dot(data.Minv, (tau - data.pinocchio.nle))
            # Computing the cost value and residuals
            pinocchio.forwardKinematics(self.state.pinocchio, data.pinocchio, q, v)
            pinocchio.updateFramePlacements(self.state.pinocchio, data.pinocchio)
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
                pinocchio.computeABADerivatives(
                    self.state.pinocchio, data.pinocchio, q, v, tau
                )
                ddq_dq = data.pinocchio.ddq_dq
                ddq_dv = data.pinocchio.ddq_dv
                data.Fx[:, :] = np.hstack([ddq_dq, ddq_dv]) + np.dot(
                    data.pinocchio.Minv, data.actuation.dtau_dx
                )
                data.Fu[:, :] = np.dot(data.pinocchio.Minv, data.actuation.dtau_du)
            else:
                pinocchio.computeRNEADerivatives(
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
        self.pinocchio = pinocchio.Model.createData(model.state.pinocchio)
        self.multibody = crocoddyl.DataCollectorMultibody(self.pinocchio)
        self.actuation = model.actuation.createData()
        self.costs = model.costs.createData(self.multibody)
        self.costs.shareMemory(self)
        self.Minv = None