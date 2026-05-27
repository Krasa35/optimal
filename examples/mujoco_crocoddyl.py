from pathlib import Path
import time
import crocoddyl
import mujoco
import mujoco.viewer
import numpy as np
import logging

from optimal.utils import setup_strict_logging
from optimal.RobotDiffModel import RobotDiffModel
from optimal.RobotMujocoModel import RobotMujocoModel

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "UR5gripper_2_finger_KKI.xml"
DT = 1e-2
HORIZON = 50
MAX_DDP_ITERS = 10
ENABLE_VERBOSE_CALLBACK = True
SHOW_PROGRESS = True
WARM_START_KP = 3.0
WARM_START_KD = 0.4

ACTUATED_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "base_to_lik",
)
START_JOINT_POSITIONS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.57,
    "elbow_joint": 1.57,
    "wrist_1_joint": 0.0,
    "wrist_2_joint": 0.0,
    "wrist_3_joint": 0.0,
    "base_to_lik": 0.0,
}
TARGET_JOINT_POSITIONS = {
    "shoulder_pan_joint": 0.3,#.2,
    "shoulder_lift_joint": 0.0,
    "elbow_joint": -0.6,#-0.6,
    "wrist_1_joint": 0.0,
    "wrist_2_joint": 0.0,
    "wrist_3_joint": 0.0,
    "base_to_lik": 0.0,
}
setup_strict_logging()


def _progress(message: str) -> None:
    if SHOW_PROGRESS:
        print(f"[progress] {message}")


def _joint_qpos_and_dof_index(model: mujoco.MjModel, joint_name: str) -> tuple[int, int]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Joint '{joint_name}' was not found in {MODEL_PATH}")
    if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError(f"Joint '{joint_name}' is not a hinge joint; this script expects 1-DoF joints.")
    q_idx = int(model.jnt_qposadr[joint_id])
    v_idx = int(model.jnt_dofadr[joint_id])
    return q_idx, v_idx


def _build_index_arrays(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    q_indices: list[int] = []
    v_indices: list[int] = []
    local_index_by_name: dict[str, int] = {}
    for local_idx, joint_name in enumerate(ACTUATED_JOINTS):
        q_idx, v_idx = _joint_qpos_and_dof_index(model, joint_name)
        q_indices.append(q_idx)
        v_indices.append(v_idx)
        local_index_by_name[joint_name] = local_idx
    return np.array(q_indices, dtype=int), np.array(v_indices, dtype=int), local_index_by_name


def _build_initial_and_target_state(
    model: mujoco.MjModel,
    q_indices: np.ndarray,
    v_indices: np.ndarray,
    local_index_by_name: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mj_data = mujoco.MjData(model)
    mujoco.mj_resetData(model, mj_data)

    q_nominal = mj_data.qpos.copy()
    v_nominal = mj_data.qvel.copy()

    q0 = q_nominal[q_indices].copy()
    v0 = v_nominal[v_indices].copy()
    x0 = np.concatenate([q0, v0])
    x_target = x0.copy()

    for joint_name, qref in TARGET_JOINT_POSITIONS.items():
        if joint_name not in local_index_by_name:
            raise ValueError(f"Target joint '{joint_name}' is not in ACTUATED_JOINTS")
        x_target[local_index_by_name[joint_name]] = qref
    return x0, x_target, q_nominal, v_nominal


def _build_joint_to_actuator_indices(
    model: mujoco.MjModel, local_index_by_name: dict[str, int]
) -> np.ndarray:
    joint_to_actuator = np.full(len(local_index_by_name), -1, dtype=int)
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            continue
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is None:
            continue
        local_idx = local_index_by_name.get(joint_name)
        if local_idx is None or joint_to_actuator[local_idx] >= 0:
            continue
        joint_to_actuator[local_idx] = actuator_id

    missing_joints = [ACTUATED_JOINTS[i] for i, actuator_id in enumerate(joint_to_actuator) if actuator_id < 0]
    if missing_joints:
        raise ValueError(
            "No actuator found for joints in ACTUATED_JOINTS: " + ", ".join(missing_joints)
        )
    return joint_to_actuator

def solve_ocp(model: mujoco.MjModel):
    _progress("Building OCP model...")
    q_indices, v_indices, local_index_by_name = _build_index_arrays(model)
    joint_to_actuator = _build_joint_to_actuator_indices(model, local_index_by_name)
    state = crocoddyl.StateVector(q_indices.size + v_indices.size)
    x0, x_target, q_nominal, v_nominal = _build_initial_and_target_state(
        model, q_indices, v_indices, local_index_by_name
    )
    _progress(
        f"State dims: nq={q_indices.size}, nv={v_indices.size}, nu={model.nu}. "
        f"Horizon={HORIZON}, max_iters={MAX_DDP_ITERS}."
    )

    running_diff = RobotDiffModel(
        model, q_indices, v_indices, q_nominal, v_nominal, state, x_target, w_q=2.0, w_v=0.01, w_u=1e-4
    )
    terminal_diff = RobotDiffModel(
        model, q_indices, v_indices, q_nominal, v_nominal, state, x_target, w_q=1000.0, w_v=2.0, w_u=0.0
    )

    running_model = crocoddyl.IntegratedActionModelEuler(running_diff, DT)
    terminal_model = crocoddyl.IntegratedActionModelEuler(terminal_diff, 0.0)

    problem = crocoddyl.ShootingProblem(x0, [running_model] * HORIZON, terminal_model)
    solver = crocoddyl.SolverFDDP(problem)
    if ENABLE_VERBOSE_CALLBACK:
        solver.setCallbacks([crocoddyl.CallbackVerbose()])

    _progress("Building warm-start rollout...")
    init_xs, init_us = _build_warm_start_trajectory(
        model, q_indices, v_indices, joint_to_actuator, x0, x_target, q_nominal, v_nominal
    )

    _progress("Starting solver...")
    solve_t0 = time.time()
    solver.th_stop = 9e-2
    ok = solver.solve(init_xs, init_us, MAX_DDP_ITERS, False, 1e-4)
    solve_dt = time.time() - solve_t0
    _progress(f"Solver finished in {solve_dt:.2f}s after {solver.iter} iterations.")
    control_norms = [float(np.linalg.norm(np.asarray(u))) for u in solver.us]
    return solver, ok, control_norms, q_indices, v_indices, q_nominal, v_nominal


def main():
    logging.info("Starting MuJoCo + Crocoddyl example...")
    MujModel = RobotMujocoModel(MODEL_PATH)
    MujModel.joint_positions = START_JOINT_POSITIONS
    #
    state = crocoddyl.StateVector(MujModel.q_indices.size + MujModel.v_indices.size)
    q_indices = MujModel.q_indices()
    v_indices = MujModel.v_indices()
    q_nominal = MujModel.joint_positions.copy()
    v_nominal = np.zeros_like(q_nominal)
    x0 = np.concatenate([q_nominal[q_indices], v_nominal[v_indices]])
    x_target = [TARGET_JOINT_POSITIONS[joint] for joint in MujModel.actuated_joint_names] + [0.0] * v_indices.size
    #
    running_diff = RobotDiffModel(
        MujModel.model, q_indices, v_indices, q_nominal, v_nominal, state, x_target, w_q=2.0, w_v=0.01, w_u=1e-4
    )
    terminal_diff = RobotDiffModel(
        MujModel.model, q_indices, v_indices, q_nominal, v_nominal, state, x_target, w_q=1000.0, w_v=2.0, w_u=0.0
    )

    running_model = crocoddyl.IntegratedActionModelEuler(running_diff, DT)
    terminal_model = crocoddyl.IntegratedActionModelEuler(terminal_diff, 0.0)

    problem = crocoddyl.ShootingProblem(x0, [running_model] * HORIZON, terminal_model)
    solver = crocoddyl.SolverFDDP(problem)
    if ENABLE_VERBOSE_CALLBACK:
        solver.setCallbacks([crocoddyl.CallbackVerbose()])

    _progress("Building warm-start rollout...")
    init_xs, init_us = _build_warm_start_trajectory(
        MujModel.model, q_indices, v_indices, joint_to_actuator, x0, x_target, q_nominal, v_nominal
    )

    _progress("Starting solver...")
    solve_t0 = time.time()
    solver.th_stop = 9e-2
    ok = solver.solve(init_xs, init_us, MAX_DDP_ITERS, False, 1e-4)
    solve_dt = time.time() - solve_t0
    _progress(f"Solver finished in {solve_dt:.2f}s after {solver.iter} iterations.")
    control_norms = [float(np.linalg.norm(np.asarray(u))) for u in solver.us]
    return solver, ok, control_norms, q_indices, v_indices, q_nominal, v_nominal


    # with mujoco.viewer.launch_passive(MujModel.model, MujModel.data) as viewer:
    #     while viewer.is_running():
    #         mujoco.mj_step(MujModel.model, MujModel.data)
    #         viewer.sync()
    #         time.sleep(0.01)



def main1() -> None:
    _progress(f"Loading MuJoCo model from {MODEL_PATH} ...")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.qpos0[:7] = [START_JOINT_POSITIONS[joint] for joint in START_JOINT_POSITIONS]
    sim_data = mujoco.MjData(model)
    print("Current sim_data.qpos:", sim_data.qpos)
    mujoco.mj_forward(model, sim_data)
    with mujoco.viewer.launch_passive(model, sim_data) as viewer:
        loop_idx = 0
        while viewer.is_running():
            print(f"[progress] Initial pose loop {loop_idx} started.")
            mujoco.mj_resetData(model, sim_data)
            mujoco.mj_forward(model, sim_data)
            viewer.sync()
            time.sleep(0.1)
    # print(sim_data.qpos)
    # for i in range(model.nu):
    #     print(f"Joint {i}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)}")
    # mujoco.mj_forward(model, sim_data)
    # print(sim_data.qpos)
    _progress("Model loaded. Solving optimal control...")
    solver, ok, control_norms, q_indices, v_indices, q_nominal, v_nominal = solve_ocp(model)
    print("solver.solve returned:", ok)
    print("Solver converged:", solver.isFeasible)
    print("Control norm range:", min(control_norms, default=0.0), "->", max(control_norms, default=0.0))
    if max(control_norms, default=0.0) < 1e-6:
        print("Warning: controls are near zero. Adjust TARGET_JOINT_POSITIONS or cost weights.")

    ctrl_min = model.actuator_ctrlrange[:, 0] if model.nu > 0 else np.zeros(0)
    ctrl_max = model.actuator_ctrlrange[:, 1] if model.nu > 0 else np.zeros(0)
    n_substeps = max(1, int(round(DT / model.opt.timestep)))

    _progress("Launching viewer and replaying trajectory in a loop...")
    with mujoco.viewer.launch_passive(model, sim_data) as viewer:
        loop_idx = 0
        progress_stride = max(1, len(solver.us) // 10)
        while viewer.is_running():
            loop_idx += 1
            if SHOW_PROGRESS:
                print(f"[progress] Playback loop {loop_idx} started.")

            mujoco.mj_resetData(model, sim_data)
            x_start = np.asarray(solver.xs[0])
            sim_data.qpos[:] = q_nominal
            sim_data.qvel[:] = v_nominal
            sim_data.qpos[q_indices] = x_start[: q_indices.size]
            sim_data.qvel[v_indices] = x_start[q_indices.size :]
            mujoco.mj_forward(model, sim_data)

            playback_t0 = time.time()
            for step_id, u in enumerate(solver.us, start=1):
                if not viewer.is_running():
                    break
                if model.nu > 0:
                    sim_data.ctrl[:] = np.clip(u, ctrl_min, ctrl_max)
                for _ in range(n_substeps):
                    mujoco.mj_step(model, sim_data)
                viewer.sync()
                time.sleep(DT)
                if SHOW_PROGRESS and (step_id % progress_stride == 0 or step_id == len(solver.us)):
                    print(f"[progress] Loop {loop_idx}: Playback {step_id}/{len(solver.us)}")

            _progress(f"Playback loop {loop_idx} finished in {time.time() - playback_t0:.2f}s.")

def main2():
    _progress(f"Loading MuJoCo model from {MODEL_PATH} ...")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    sim_data = mujoco.MjData(model)

    # 1. Set the positions safely using the joint names
    for joint_name, target_pos in START_JOINT_POSITIONS.items():
        try:
            # Find the ID of the joint
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id != -1:
                # Find where this joint's data lives in the qpos array
                qpos_adr = model.jnt_qposadr[joint_id]
                
                # Update both the current position and the controller target
                sim_data.qpos[qpos_adr] = target_pos
                
                # If you have actuators with the same name, set their targets too
                actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
                if actuator_id != -1:
                    sim_data.ctrl[actuator_id] = target_pos
        except Exception as e:
            print(f"Error setting {joint_name}: {e}")

    print("Target initialized. Current sim_data.qpos:", sim_data.qpos)

    # 2. Forward kinematics to update the 3D geometry with the new positions
    mujoco.mj_forward(model, sim_data)

    # 3. Viewer loop
    with mujoco.viewer.launch_passive(model, sim_data) as viewer:
        while viewer.is_running():
            # DO NOT call mj_resetData here! It will wipe your custom positions.
            
            # Step the physics engine forward
            mujoco.mj_step(model, sim_data)
            
            # Sync the viewer to the new physics state
            viewer.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    main()
