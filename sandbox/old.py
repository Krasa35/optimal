def _build_warm_start_trajectory(
    model: mujoco.MjModel,
    q_indices: np.ndarray,
    v_indices: np.ndarray,
    joint_to_actuator: np.ndarray,
    x0: np.ndarray,
    x_target: np.ndarray,
    q_nominal: np.ndarray,
    v_nominal: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    nq = q_indices.size
    sim_data = mujoco.MjData(model)
    mujoco.mj_resetData(model, sim_data)
    sim_data.qpos[:] = q_nominal
    sim_data.qvel[:] = v_nominal
    sim_data.qpos[q_indices] = x0[:nq]
    sim_data.qvel[v_indices] = x0[nq:]
    mujoco.mj_forward(model, sim_data)

    ctrl_min = model.actuator_ctrlrange[:, 0] if model.nu > 0 else np.zeros(0)
    ctrl_max = model.actuator_ctrlrange[:, 1] if model.nu > 0 else np.zeros(0)
    has_ctrl_limits = bool(np.any(model.actuator_ctrllimited)) if model.nu > 0 else False
    n_substeps = max(1, int(round(DT / model.opt.timestep)))

    warm_xs = [x0.copy()]
    warm_us: list[np.ndarray] = []
    q_start = x0[:nq]
    q_goal = x_target[:nq]

    for k in range(HORIZON):
        alpha = (k + 1) / HORIZON
        q_des = (1.0 - alpha) * q_start + alpha * q_goal
        q_curr = sim_data.qpos[q_indices]
        v_curr = sim_data.qvel[v_indices]
        tau_local = WARM_START_KP * (q_des - q_curr) - WARM_START_KD * v_curr

        u = np.zeros(model.nu)
        for local_idx, actuator_id in enumerate(joint_to_actuator):
            u[actuator_id] = tau_local[local_idx]
        if has_ctrl_limits:
            u = np.clip(u, ctrl_min, ctrl_max)

        warm_us.append(u.copy())
        if model.nu > 0:
            sim_data.ctrl[:] = u
        for _ in range(n_substeps):
            mujoco.mj_step(model, sim_data)

        x_next = np.concatenate([sim_data.qpos[q_indices].copy(), sim_data.qvel[v_indices].copy()])
        if not np.all(np.isfinite(x_next)):
            raise ValueError("Warm-start rollout produced non-finite states.")
        warm_xs.append(x_next)

    return warm_xs, warm_us