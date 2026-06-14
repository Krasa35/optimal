import os
import time
import numpy as np
import pinocchio as pin
import pybullet as p
import pybullet_data
import crocoddyl

def main():
    # =========================================================================
    # CZĘŚĆ 1: OPTYMALIZACJA TRAJEKTORII W CROCODDYLU (MÓZG)
    # =========================================================================
    print("--- ROZPOCZĘCIE FAZY PLANOWANIA (CROCODDYL) ---")
    
    urdf_path = "../models/ur10/ur10.urdf" 
    
    # Wczytanie modelu z ustawieniem grawitacji
    pin_model = pin.buildModelFromUrdf(urdf_path)
    pin_model.gravity = pin.Motion(np.array([0, 0, -9.81, 0, 0, 0]))
    
    # Przygotowanie modeli Crocoddyla
    state = crocoddyl.StateMultibody(pin_model)
    actuation = crocoddyl.ActuationModelFull(state)
    nv, nu = state.nv, state.nv
    dt_plan = 0.01 # Krok czasowy 10ms (100 Hz)
    T = 100        # Horyzont 150 kroków = 1.5 sekundy lotu
    
    # Start z pozycji bezpiecznej (lekko ugiętej, żeby uniknąć osobliwości)
    q0 = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    x0 = np.concatenate([q0, np.zeros(nv)])
    
    # Automatyczne szukanie efektora (w UR10 to zazwyczaj 'tool0' lub 'wrist_3_link')
    ee_frame_name = "wrist_3_link"
    if not pin_model.existFrame(ee_frame_name):
        ee_frame_name = "tool0"
    target_id = pin_model.getFrameId(ee_frame_name)
    
    # Cel: przemieścić końcówkę w konkretne miejsce XYZ
    target_pos = np.array([0.6, 0.3, 0.4]) 
    target_rot = np.eye(3) # Orientacja nas teraz mniej interesuje, skupmy się na pozycji
    
    # --- BUDOWANIE PROBLEMU OPTYMALIZACJI ---
    # 1. Błąd pozycji końcówki (Residual)
    eePoseResidual = crocoddyl.ResidualModelFramePlacement(
        state, target_id, pin.SE3(target_rot, target_pos), nu
    )
    
    # 2. Koszty (Kary za złe zachowanie)
    xResidual = crocoddyl.ResidualModelState(state, x0, nu)
    uResidual = crocoddyl.ResidualModelJointEffort(state, actuation, nu)
    
    eePoseCost = crocoddyl.CostModelResidual(state, eePoseResidual)
    xRegCost = crocoddyl.CostModelResidual(state, xResidual)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    
    runningCosts = crocoddyl.CostModelSum(state)
    terminalCosts = crocoddyl.CostModelSum(state)
    
    # Podczas lotu: lekko karzemy za zużycie prądu (u) i zbytnie odchylenia stanu (x)
    runningCosts.addCost("uReg", uRegCost, 1e-4) # Niska waga, bo grawitacja dla UR10 wymaga dużych momentów!
    runningCosts.addCost("xReg", xRegCost, 1e-3)
    # runningCosts.addCost("eeTracking", eePoseCost, 1e-1) # Opcjonalnie: każ podążać do celu już w trakcie lotu
    
    # Na samym końcu: BEZWZGLĘDNIE osiągnij cel (ogromna waga)
    terminalCosts.addCost("goalPose", eePoseCost, 1e4)
    
    # Modele dynamiki
    runningModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCosts), dt_plan
    )
    terminalModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCosts), 0.0
    )
    
    # --- ROZWIĄZYWANIE (FDDP) ---
    problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
    solver = crocoddyl.SolverFDDP(problem)
    solver.setCallbacks([crocoddyl.CallbackVerbose()])
    
    # Warm-start (dajemy mu x0 i zera dla u, jako punkt wyjścia)
    xs_init = [x0] * (T + 1)
    us_init = [np.zeros(nu)] * T
    
    print("Rozpoczynam optymalizację trajektorii. Poczekaj chwilę...")
    solver.solve(xs_init, us_init, maxiter=200)
    
    # Zapisujemy nasz "Przepis na ruch"
    xs_plan = solver.xs
    us_plan = solver.us
    print(f"Plan gotowy! Wygenerowano {len(us_plan)} kroków sterowania.")

    # =========================================================================
    # CZĘŚĆ 2: WYKONANIE W PYBULLET (CIAŁO + FEEDBACK)
    # =========================================================================
    print("\n--- ROZPOCZĘCIE FAZY WYKONANIA (PYBULLET) ---")
    
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.81)
    
    # Ustawiamy PyBullet na ten sam krok czasowy co Crocoddyl dla perfekcyjnej synchronizacji
    p.setTimeStep(dt_plan)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.5])

    # Ustawiamy robota w pozycji x0 (początek trajektorii Crocoddyla)
    robot_id = p.loadURDF(urdf_path, useFixedBase=True)
    
    active_pb_joints = []
    for j in range(p.getNumJoints(robot_id)):
        if p.getJointInfo(robot_id, j)[2] != p.JOINT_FIXED:
            active_pb_joints.append(j)
            p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, force=0) # Wyłączamy sztywność
            
    # Ręcznie przestawiamy robota w pozycję startową
    for i, pb_j in enumerate(active_pb_joints):
        p.resetJointState(robot_id, pb_j, q0[i])

    # Rysujemy małą czerwoną sferę w punkcie celu, żeby widzieć, gdzie robot leci
    target_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 0.7])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=target_visual, basePosition=target_pos)

    # Delikatne wzmocnienia do korygowania rozbieżności symulatorów
    kp_fb = np.array([20.0, 40.0, 20.0, 5.0, 2.0, 5.0])
    kd_fb = np.array([2.0,  4.0,  2.0,  1.0,  1.0,  0.5])
    tau_limit = np.array([330.0, 330.0, 150.0, 56.0, 56.0, 56.0])

    print("Zaczynamy ruch za 2 sekundy...")
    time.sleep(2.0)

    for i in range(T):
        # 1. Odczyt stanu rzeczywistego
        q_real = np.zeros(nv)
        v_real = np.zeros(nv)
        for idx, pb_j in enumerate(active_pb_joints):
            pos, vel, _, _ = p.getJointState(robot_id, pb_j)
            q_real[idx] = pos
            v_real[idx] = vel

        # 2. Pobranie planu
        q_ideal = xs_plan[i][:nv]
        v_ideal = xs_plan[i][nv:]
        tau_crocoddyl = us_plan[i] # To już zawiera kompensację grawitacji!
        
        # 3. Dodanie stabilizatora (Korekcja)
        tau_feedback = kp_fb * (q_ideal - q_real) + kd_fb * (v_ideal - v_real)
        tau_feedback = np.zeros(nu) # Opcjonalnie: wyłącz korekcję, żeby zobaczyć samą otwartą pętlę Crocoddyla
        
        # 4. Aplikacja siły
        tau_cmd = np.clip(tau_crocoddyl + tau_feedback, -tau_limit, tau_limit)
        
        for j_idx, pb_joint in enumerate(active_pb_joints):
            p.setJointMotorControl2(robot_id, pb_joint, p.TORQUE_CONTROL, force=tau_cmd[j_idx])
            
        p.stepSimulation()
        time.sleep(dt_plan) # Żebyśmy widzieli ruch w czasie rzeczywistym

    print("Trajektoria zakończona!")
    input("Naciśnij Enter, aby zamknąć...")
    p.disconnect()

if __name__ == '__main__':
    main()
