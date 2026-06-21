import numpy as np
import matplotlib.pyplot as plt


def thermal_energy(us: np.ndarray, dt: float, plot: bool = False) -> float:
    """
    Integral of squared torques — proportional to Joule heating losses.
        E_thermal ∝ ∫ Σ τ_i(t)² dt
    Args:
        us: (T, nu) array of torques at each timestep.
        dt: timestep in seconds.
        plot: whether to plot the thermal energy over time.
    Returns:
        Scalar thermal energy proxy [N²m²·s].
    """
    us = np.asarray(us)
    if plot == True:
        energy_per_step = np.sum(us ** 2, axis=1) * dt
        plt.plot(energy_per_step)
        plt.xlabel("Timestep")
        plt.ylabel("Thermal energy increment (N²m²·s)")
        plt.title("Thermal energy per timestep")
        plt.grid()
        plt.show()
    return float(np.sum(us ** 2) * dt)

def mechanical_work(us: np.ndarray, xs: np.ndarray, dt: float, plot: bool = False) -> float:
    """
    Absolute mechanical work — accounts for both acceleration and braking.
        E_mech = ∫ Σ |τ_i(t) · q̇_i(t)| dt
    Args:
        us: (T, nu) array of torques.
        xs: (T, 2*nu) state array [q(nu), v(nu)] as returned by visualize().
        dt: timestep in seconds.
        plot: whether to plot the mechanical work over time.
    Returns:
        Scalar mechanical work [J].
    """
    us = np.asarray(us)
    xs = np.asarray(xs)
    nu = us.shape[1]
    v = xs[:, nu:2*nu]   # velocity part of state
    if plot == True:
        work_per_step = np.sum(np.abs(us * v), axis=1) * dt
        plt.plot(work_per_step)
        plt.xlabel("Timestep")
        plt.ylabel("Mechanical work increment (J)")
        plt.title("Mechanical work per timestep")
        plt.grid()
        plt.show()
    return float(np.sum(np.abs(us * v)) * dt)

def peak_power(us: np.ndarray, xs: np.ndarray, plot: bool = False) -> float:
    """
    Maximum instantaneous total power — critical for power supply sizing.
        P_peak = max_t ( Σ |τ_i(t) · q̇_i(t)| )
    Args:
        us: (T, nu) array of torques.
        xs: (T, 2*nu) state array [q(nu), v(nu)] as returned by visualize().
        plot: whether to plot the peak power over time.
    Returns:
        Scalar peak power [W].
    """
    us = np.asarray(us)
    xs = np.asarray(xs)
    nu = us.shape[1]
    v = xs[:, nu:2*nu]
    power_per_step = np.sum(np.abs(us * v), axis=1)
    if plot == True:
        plt.plot(power_per_step)
        plt.xlabel("Timestep")
        plt.ylabel("Total power (W)")
        plt.title("Total power per timestep")
        plt.grid()
        plt.show()
    return float(np.max(power_per_step))

def terminal_error(q_final: np.ndarray, q_target: np.ndarray) -> float:
    """
    Distance from final position to target.
        error = ||q_final - q_target||
    Args:
        q_final: (nu,) final joint positions.
        q_target: (nu,) target joint positions.
    Returns:
        Scalar terminal error [rad].
    """
    return float(np.linalg.norm(q_final[:3] - q_target[:3]) + np.linalg.norm(q_final[3:] - q_target[3:]))

def tracking_error(xs: np.ndarray, targets, segment_lengths, degrees: bool = True, plot: bool = False) -> float:
    """
    Settling precision — mean terminal joint error across all path segments.
    For each segment the realized joint configuration at the segment's final
    timestep is compared against that segment's target using terminal_error
    (body joints fully weighted, wrist joints down-weighted).
        precision = mean_k terminal_error(q_end_k, q_target_k)
    Args:
        xs: (T, 2*nu) realized state trajectory [q(nu), v(nu)].
        targets: list of target joint configurations, one per segment.
        segment_lengths: list of int, number of timesteps per segment.
        degrees: return the error in degrees (True) or radians (False).
    Returns:
        Scalar mean terminal error [deg or rad].
    """
    xs = np.asarray(xs)
    nq = xs.shape[1] // 2
    idx = 0
    errs = []
    for target, n in zip(targets, segment_lengths):
        idx += n
        errs.append(terminal_error(xs[idx - 1, :nq], np.asarray(target)[:nq]))
    err = float(np.mean(errs))
    if plot:
        plt.plot(errs)
        plt.xlabel("Segment")
        plt.ylabel("Terminal error (rad)" if not degrees else "Terminal error (deg)")
        plt.title("Tracking error per segment")
        plt.grid()
        plt.show()
    return err * 180.0 / np.pi if degrees else err

def torque_jerk(us: np.ndarray, plot: bool = False) -> float:
    """
    Sum of squared changes in torque — a smoothness metric.
        jerk = ∫ Σ (dτ_i/dt)² dt
    Args:
        us: (T, nu) array of torques.
        plot: whether to plot the torque jerk over time.
    Returns:
        Scalar torque jerk [N²m²/s].
    """
    us = np.asarray(us)
    jerk_per_step = np.sum(np.diff(us, axis=0) ** 2, axis=1)
    if plot == True:
        plt.plot(jerk_per_step)
        plt.xlabel("Timestep")
        plt.ylabel("Torque jerk increment (N²m²/s)")
        plt.title("Torque jerk per timestep")
        plt.grid()
        plt.show()
    return float(np.sum(jerk_per_step))

def plot_metrics(us: np.ndarray, xs: np.ndarray, dt: float, targets, segment_lengths, controller_time: float = None):
    us = np.asarray(us)
    xs = np.asarray(xs)
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    fig.suptitle('Optimal Control Metrics', fontsize=16)
    
    ax = axes[0, 0]
    energy_per_step = np.sum(us ** 2, axis=1) * dt
    ax.plot(energy_per_step)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Thermal energy increment (N²m²·s)")
    ax.set_title("Thermal energy per timestep")
    ax.grid()
    
    ax = axes[0, 1]
    nu = us.shape[1]
    v = xs[:, nu:2*nu]
    work_per_step = np.sum(np.abs(us * v), axis=1) * dt
    ax.plot(work_per_step)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mechanical work increment (J)")
    ax.set_title("Mechanical work per timestep")
    ax.grid()
    
    ax = axes[1, 0]
    power_per_step = np.sum(np.abs(us * v), axis=1)
    ax.plot(power_per_step)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Total power (W)")
    ax.set_title("Total power per timestep")
    ax.grid()
    
    ax = axes[1, 1]
    jerk_per_step = np.sum(np.diff(us, axis=0) ** 2, axis=1)
    ax.plot(jerk_per_step)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Torque jerk increment (N²m²/s)")
    ax.set_title("Torque jerk per timestep")
    ax.grid()
    
    ax = axes[2, 0]
    nq = xs.shape[1] // 2
    idx = 0
    errs = []
    for target, n in zip(targets, segment_lengths):
        idx += n
        errs.append(terminal_error(xs[idx - 1, :nq], np.asarray(target)[:nq]))
    ax.plot(errs)
    ax.set_xlabel("Segment")
    ax.set_ylabel("Terminal error (deg)")
    ax.set_title("Tracking error per segment")
    ax.grid()
    
    axes[2, 1].axis('off')

    # Add statistics text box
    stats_text = f"Total Thermal Energy:   {thermal_energy(us, dt):.2f} N²m²·s\n"
    stats_text += f"Total Mechanical Work:  {mechanical_work(us, xs, dt):.2f} J\n"
    stats_text += f"Peak Power:             {peak_power(us, xs):.2f} W\n"
    stats_text += f"Mean Tracking Error:    {tracking_error(xs, targets, segment_lengths):.2f} deg\n"
    if controller_time is not None:
        stats_text += f"Control Computation Time: {controller_time:.4f} s\n"
    stats_text += f"Total Torque Jerk:      {torque_jerk(us):.2f} N²m²/s"
    axes[2, 1].text(0.1, 0.5, stats_text, fontsize=12, ha='left', va='center', transform=axes[2, 1].transAxes)
    
    plt.tight_layout()
    plt.show()

def summary_metrics(us: np.ndarray, xs: np.ndarray, dt: float, targets, segment_lengths) -> dict:
    metrics = {
        "thermal_energy": thermal_energy(us, dt),
        "mechanical_work": mechanical_work(us, xs, dt),
        "peak_power": peak_power(us, xs),
        "tracking_error": tracking_error(xs, targets, segment_lengths),
        "torque_jerk": torque_jerk(us)
    }
    plot_metrics(us, xs, dt, targets, segment_lengths)
    return metrics