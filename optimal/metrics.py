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
    return float(np.linalg.norm(q_final[:3] - q_target[:3]) + 0.1*np.linalg.norm(q_final[3:] - q_target[3:]))

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