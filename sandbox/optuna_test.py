import optuna
import numpy as np
import pinocchio as pin
import mujoco

# Import your existing controller
from optimal.RobotController import RobotController

# --- 1. Setup ---
# Load your model (just like in your notebook)
mjcf_path = "../models/ur10/ur10_sandbox.xml"
urdf_path = "../models/ur10/ur10_sandbox.urdf"
robot_mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
robot_pin_model = pin.buildModelFromUrdf(urdf_path)

q_start = pin.neutral(robot_pin_model)
q_target = np.array([0.4, -0.4, 1.5, -0.8, 1.5, 0.5, 0.0, 0.0])
DT = robot_mj_model.opt.timestep
T = 500

# --- 2. Define the Optuna Objective ---
def objective(trial):
    # Ask Optuna to guess the weights (we use log scale because weights can vary wildly)
    track_weight = trial.suggest_float("track_weight", 1e-1, 1e5, log=True)
    ctrl_weight = trial.suggest_float("ctrl_weight", 1e-5, 1e2, log=True)
    terminal_weight = trial.suggest_float("terminal_weight", 1e2, 1e7, log=True)
    
    # Initialize your controller
    controller = RobotController(robot_pin_model)
    
    try:
        # NOTE: You must modify your `compute_control` method to accept these weights!
        xs, us = controller.compute_control(
            q_start=q_start, 
            q_target=q_target, 
            T=T, 
            DT=DT, 
            option="boxfddp",
            track_weight=track_weight,         # Pass Optuna's guesses here
            ctrl_weight=ctrl_weight,
            terminal_weight=terminal_weight
        )
    except Exception as e:
        # If the solver crashes (e.g. weights were numerically unstable), 
        # return a massive penalty so Optuna knows it was a bad guess.
        return float('inf')

    # --- 3. Evaluate the Metric (How good was the result?) ---
    
    # A. Terminal Error (Distance from final position to target)
    q_final = xs[-1][:robot_pin_model.nq] # Get last joint position
    terminal_error = np.linalg.norm((q_final - q_target)) 
    
    # B. Control Effort (Sum of squared torques)
    us_array = np.array(us)
    control_effort = np.sum(us_array ** 2)
    
    # C. Smoothness (Sum of squared changes in torque)
    torque_jerk = np.sum(np.diff(us_array, axis=0) ** 2)
    
    # Combine into a single score. 
    # Notice we weigh terminal error heavily because missing the target is a failure.
    metric_score = (100.0 * terminal_error) + (0.01 * control_effort) + (0.1 * torque_jerk)
    
    return metric_score

# --- 4. Run the Optimization ---
if __name__ == "__main__":
    # Create the study. We want to MINIMIZE the metric_score.
    study = optuna.create_study(direction="minimize")
    
    # Run 100 trials (guesses)
    print("Starting Optuna search...")
    study.optimize(objective, n_trials=100)
    
    # Print the best result
    print("\n=== Best Coefficients Found ===")
    print(f"Tracking Weight: {study.best_params['track_weight']}")
    print(f"Control Weight: {study.best_params['ctrl_weight']}")
    print(f"Terminal Tracking Weight: {study.best_params['terminal_weight']}")
    print(f"Best Metric Score: {study.best_value}")