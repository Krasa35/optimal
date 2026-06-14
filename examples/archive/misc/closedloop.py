import pybullet as p
import pybullet_data
import time
import numpy as np
import pinocchio as pin

def compute_open_loop_plan(model, data, q_start, q_target, horizon, dt, kp, kd):
    q = q_start.copy()
    v = np.zeros(model.nv)
    e_prev = np.zeros(model.nq) # Startujemy z zerowym błędem
    
    xs = [np.concatenate([q, v])]
    us = []
    
    # 1. FIZYCZNE LIMITY SILNIKÓW UR10 (Bardzo ważne!)
    tau_max = np.array([330.0, 330.0, 150.0, 56.0, 56.0, 56.0])
    
    for step in range(horizon):
        # 2. GENERATOR TRAJEKTORII (Gładkie przesuwanie celu)
        # Cel (q_ref) płynnie wędruje od q_start do q_target przez 80% czasu symulacji.
        alpha = min(1.0, step / (horizon * 0.8))
        q_ref = q_start + alpha * (q_target - q_start)
        
        # Błąd liczony jest do tego PŁYNNEGO punktu (jest zawsze bardzo mały)
        err = q_ref - q
        derr = (err - e_prev) / dt
        e_prev = err
        
        # Obliczenie momentów
        u = kp * err + kd * derr
        g = pin.computeGeneralizedGravity(model, data, q)
        tau = u + g
        
        # 3. NASYCENIE (TORQUE CLIPPING)
        # Odcina wszystkie kosmiczne wartości do bezpiecznych limitów robota
        tau = np.clip(tau, -tau_max, tau_max)
        
        # Krok wirtualnej fizyki Pinocchio
        a = pin.aba(model, data, q, v, tau)
        v = v + dt * a
        q = pin.integrate(model, q, dt * v)
        
        us.append(tau.copy())
        xs.append(np.concatenate([q.copy(), v.copy()]))
        
    return xs, us

def main():
    urdf_path = "../models/ur10/ur10.urdf" 
    dt = 1.0 / 240.0  # Krok czasowy (standardowy dla PyBullet)
    horizon = 800     # Czas symulacji: 800 kroków * dt ≈ 3.3 sekundy
    
    # =========================================================================
    # ETAP 1: GENEROWANIE PLANU W PINOCCHIO (OTWARTA PĘTLA)
    # =========================================================================
    print("Rozpoczęcie planowania trajektorii w Pinocchio...")
    pin_model = pin.buildModelFromUrdf(urdf_path)
    pin_model.gravity = pin.Motion(np.array([0, 0, -9.81, 0, 0, 0]))
    pin_data = pin_model.createData()
    
    q_start = pin.neutral(pin_model)
    q_target = np.array([0.7, -1.2, 1.5, -0.8, 1.5, 0.5]) # Punkt docelowy
    
    # Nastawy wirtualnego regulatora
    kp = np.array([400.0, 800.0, 400.0, 100.0, 40.0, 10.0])
    kd = np.array([40.0,  80.0,  40.0,  10.0,  4.0,  1.0])
    
    # Wyliczamy gotowy "przepis na momenty sił" (us)
    xs_ideal, us_plan = compute_open_loop_plan(pin_model, pin_data, q_start, q_target, horizon, dt, kp, kd)
    print(f"Plan wygenerowany! Zapisano {len(us_plan)} kroków sterowania momentowego.")

# =========================================================================
    # ETAP 2: ODTWARZANIE TRAJEKTORII (FEEDFORWARD + FEEDBACK)
    # =========================================================================
    print("\nUruchamianie świata fizycznego PyBullet...")
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(dt)
    
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=135, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.5])

    robot_id = p.loadURDF(urdf_path, useFixedBase=True)

    # Mapowanie przegubów i wyłączenie domyślnych silników pozycyjnych PyBullet
    active_pb_joints = []
    for j in range(p.getNumJoints(robot_id)):
        if p.getJointInfo(robot_id, j)[2] != p.JOINT_FIXED:
            active_pb_joints.append(j)
            p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, force=0)

    print("Odtwarzanie stabilizowane sprzężeniem zwrotnym za 2 sekundy...")
    time.sleep(2.0)
    
    # !!! NOWE: Wzmocnienia korekcyjne dla PyBullet (Feedback)
    # Służą wyłącznie do tłumienia małych odchyłek od idealnego planu
    kp_fb = np.array([200.0, 400.0, 200.0, 50.0, 20.0, 5.0])
    kd_fb = np.array([20.0,  40.0,  20.0,  5.0,  2.0,  0.5])
    
    # Pętla wykonawcza ze sprzężeniem zwrotnym
    for i in range(horizon):
        # 1. Odczyt RZECZYWISTEGO stanu z fizyki PyBullet
        q_real = np.zeros(pin_model.nq)
        v_real = np.zeros(pin_model.nv)
        for idx, pb_j in enumerate(active_pb_joints):
            pos, vel, _, _ = p.getJointState(robot_id, pb_j)
            q_real[idx] = pos
            v_real[idx] = vel
            
        # 2. Pobranie IDEALNEGO (zaplanowanego) stanu i momentu z Pinocchio
        x_ideal = xs_ideal[i]
        q_ideal = x_ideal[:pin_model.nq]
        v_ideal = x_ideal[pin_model.nq:]
        
        tau_feedforward = us_plan[i] # To jest nasz ślepy plan momentów
        
        # 3. Obliczenie poprawki sprzężenia zwrotnego (Feedback)
        # Jeśli q_real == q_ideal, ta część wynosi zero. Działa tylko przy odchyłkach!
        err = q_ideal - q_real
        derr = v_ideal - v_real
        tau_feedback = kp_fb * err + kd_fb * derr
        
        tau_cmd = tau_feedforward + tau_feedback
        
        # Zabezpieczenie przed skokami i sprzężeniem zwrotnym z PyBullet!
        tau_max = np.array([330.0, 330.0, 150.0, 56.0, 56.0, 56.0])
        tau_cmd = np.clip(tau_cmd, -tau_max, tau_max) 
        
        # Aplikujemy siłę do silników
        for j_idx, pb_joint in enumerate(active_pb_joints):
            p.setJointMotorControl2(robot_id, pb_joint, p.TORQUE_CONTROL, force=tau_cmd[j_idx])
            
        p.stepSimulation()
        time.sleep(dt)

    print("Odtwarzanie zakończone pomyślnie i stabilnie.")
    input("Naciśnij Enter, aby zamknąć symulator...")
    p.disconnect()

if __name__ == '__main__':
    main()