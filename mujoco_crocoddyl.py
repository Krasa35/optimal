import numpy as np
import time
import mujoco
import mujoco.viewer
import crocoddyl

# ===== 1) Load MuJoCo model =====
model = mujoco.MjModel.from_xml_path("models/UR5gripper_2_finger_KKI.xml")
sim_data = mujoco.MjData(model)

# ===== 2) Crocoddyl state/action dims =====
state = crocoddyl.StateVector(model.nq + model.nv)  # [q, v]
actuation = crocoddyl.ActuationModelFull(state)
nu = actuation.nu
dt = 1e-2

# Target state: upright pendulum, zero velocity
x_target = np.array([np.pi, 0.0], dtype=float)


class PendulumDiffModel(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, w_goal=1.0, w_u=1e-4):
        # nr=1 is enough here (scalar cost)
        super().__init__(state, nu, 1)
        self.mj_data = mujoco.MjData(model)
        self.w_goal = w_goal
        self.w_u = w_u

    def calc(self, data, x, u=None):
        # terminal node may call calc(data, x) with no u
        if u is None:
            u = np.zeros(nu)

        # Set MuJoCo state/control
        self.mj_data.qpos[:] = x[:model.nq]
        self.mj_data.qvel[:] = x[model.nq:]
        self.mj_data.ctrl[:] = u

        # Forward dynamics -> qacc
        mujoco.mj_forward(model, self.mj_data)

        # Differential model output xdot = [qdot, vdot]
        qdot = self.mj_data.qvel.copy()
        vdot = self.mj_data.qacc.copy()
        data.xout = np.concatenate([qdot, vdot])

        # Scalar cost
        dx = x - x_target
        data.cost = 0.5 * self.w_goal * float(dx @ dx) + 0.5 * self.w_u * float(u @ u)

    def calcDiff(self, data, x, u=None):
        # terminal node may call calcDiff(data, x) with no u
        if u is None:
            u = np.zeros(nu)
        # Leave empty: NumDiff wrapper computes derivatives numerically
        pass


# ===== 3) Running & terminal models =====
running_diff = PendulumDiffModel(w_goal=1.0, w_u=1e-4)
terminal_diff = PendulumDiffModel(w_goal=100.0, w_u=0.0)

running_numdiff = crocoddyl.DifferentialActionModelNumDiff(running_diff, False)
terminal_numdiff = crocoddyl.DifferentialActionModelNumDiff(terminal_diff, False)

running_model = crocoddyl.IntegratedActionModelEuler(running_numdiff, dt)
terminal_model = crocoddyl.IntegratedActionModelEuler(terminal_numdiff, 0.0)

# ===== 4) Solve OCP =====
T = 100
x0 = np.array([0.0, 0.0], dtype=float)

problem = crocoddyl.ShootingProblem(x0, [running_model] * T, terminal_model)
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([crocoddyl.CallbackVerbose()])

ok = ddp.solve([], [], 100)
print("ddp.solve returned:", ok)
print("Solver converged:", ddp.isFeasible)

# ===== 5) Visualize in MuJoCo =====
if ddp.isFeasible:
    with mujoco.viewer.launch_passive(model, sim_data) as viewer:
        mujoco.mj_resetData(model, sim_data)
        sim_data.qpos[:] = ddp.xs[0][:model.nq]
        sim_data.qvel[:] = ddp.xs[0][model.nq:]
        mujoco.mj_forward(model, sim_data)

        for u in ddp.us:
            if not viewer.is_running():
                break
            sim_data.ctrl[:] = u
            mujoco.mj_step(model, sim_data)
            viewer.sync()
            time.sleep(dt)
        input("Press Enter to exit...")
        
else:
    print("DDP did not converge. Try increasing T (e.g. 150-250) or maxiter.")
