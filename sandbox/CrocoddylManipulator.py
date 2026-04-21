import crocoddyl
import example_robot_data
import numpy as np
import pinocchio
import meshcat.geometry as g

robot = example_robot_data.load("talos_arm")
robot_model = robot.model

DT = 1e-3
T = 25
target = np.array([0.4, 0.0, 0.4])

display = crocoddyl.MeshcatDisplay(robot)
display.robot.viewer["world/point"].set_object(g.Sphere(0.05))
display.robot.viewer["world/point"].set_transform(
    np.array(
        [
            [1.0, 0.0, 0.0, target[0]],
            [0.0, 1.0, 0.0, target[1]],
            [0.0, 0.0, 1.0, target[2]],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
)

# Create the cost functions
state = crocoddyl.StateMultibody(robot.model)
goalTrackingCost = crocoddyl.CostModelResidual(
    state,
    crocoddyl.ResidualModelFrameTranslation(
        state, robot_model.getFrameId("gripper_left_joint"), target
    ),
)
xRegCost = crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelState(state))
uRegCost = crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelControl(state))

# Create cost model per each action model
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1e2)
runningCostModel.addCost("stateReg", xRegCost, 1e-4)
runningCostModel.addCost("ctrlReg", uRegCost, 1e-7)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e5)
terminalCostModel.addCost("stateReg", xRegCost, 1e-4)
terminalCostModel.addCost("ctrlReg", uRegCost, 1e-7)

# Create the actuation model
actuationModel = crocoddyl.ActuationModelFull(state)

# Create the action model
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuationModel, runningCostModel
    ),
    DT,
)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuationModel, terminalCostModel
    )
)
# runningModel.differential.armature = 0.2 * np.ones(state.nv)
# terminalModel.differential.armature = 0.2 * np.ones(state.nv)

# Create the problem
q0 = np.array([2.0, 1.5, -2.0, 0.0, 0.0, 0.0, 0.0])
x0 = np.concatenate([q0, pinocchio.utils.zero(state.nv)])
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving it with the DDP algorithm
ddp.solve()

# Visualizing the solution in gepetto-viewer
display.displayFromSolver(ddp)

robot_data = robot_model.createData()
xT = ddp.xs[-1]
pinocchio.forwardKinematics(robot_model, robot_data, xT[: state.nq])
pinocchio.updateFramePlacements(robot_model, robot_data)
print(
    "Finally reached = ",
    robot_data.oMf[robot_model.getFrameId("gripper_left_joint")].translation.T,
)


import numpy as np
import example_robot_data

talos_arm = example_robot_data.load("talos_arm")
robot_model = talos_arm.model  # getting the Pinocchio model

# Defining a initial state
q0 = np.array([0.173046, 1.0, -0.52366, 0.0, 0.0, 0.1, -0.005])
x0 = np.concatenate([q0, np.zeros(talos_arm.model.nv)])

# Create the cost functions
target = np.array([0.4, 0.0, 0.4])
state = crocoddyl.StateMultibody(robot_model)
frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
    state, robot_model.getFrameId("gripper_left_joint"), target
)
goalTrackingCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)
xRegCost = crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelState(state))
uRegCost = crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelControl(state))

# Create cost model per each action model
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1e2)
runningCostModel.addCost("stateReg", xRegCost, 1e-4)
runningCostModel.addCost("ctrlReg", uRegCost, 1e-7)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e5)
terminalCostModel.addCost("stateReg", xRegCost, 1e-4)
terminalCostModel.addCost("ctrlReg", uRegCost, 1e-7)

# Running and terminal action models
DT = 1e-3
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
     

# For this optimal control problem, we define 250 knots (or running action
# models) plus a terminal knot
T = 250
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
solver = crocoddyl.SolverFDDP(problem)
log = crocoddyl.CallbackLogger()

# Using the meshcat displayer, you could enable gepetto viewer for nicer view
# display = crocoddyl.GepettoDisplay(talos_arm, 4, 4)
display = crocoddyl.MeshcatDisplay(talos_arm, 4, 4)
solver.setCallbacks([log, crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])


# Emdebbed meshcat in this cell
display.robot.viewer.jupyter_cell()


# Solving it with the DDP algorithm
solver.solve()

# Printing the reached position
frame_idx = talos_arm.model.getFrameId("gripper_left_joint")
xT = solver.xs[-1]
qT = xT[:talos_arm.model.nq]
print()
print("The reached pose by the wrist is")
print(talos_arm.framePlacement(qT, frame_idx))


%matplotlib inline
# # Plotting the solution and the DDP convergence
crocoddyl.plotOCSolution(log.xs, log.us)
crocoddyl.plotConvergence(
    log.costs, log.pregs, log.dregs, log.grads, log.stops, log.steps
)

# Visualizing the solution in gepetto-viewer
display.displayFromSolver(solver)