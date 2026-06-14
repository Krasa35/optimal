import pybullet as p
import pybullet_data
import time
import numpy as np
import pinocchio as pin

def main():
    # Ścieżka do Twojego lokalnego pliku URDF
    urdf_path = "../models/ur10/ur10.urdf" 
    
    # 1. INICJALIZACJA PINOCCHIO (Mózg matematyczny)
    print("Wczytywanie modelu do Pinocchio...")
    pin_model = pin.buildModelFromUrdf(urdf_path)
    # KLUCZOWE: Musimy jawnie ustawić wektor grawitacji w Pinocchio!
    pin_model.gravity = pin.Motion(np.array([0, 0, -9.81, 0, 0, 0]))
    pin_data = pin_model.createData()
    
    # Definiujemy punkt startowy (wszystko na zero) i punkt docelowy (q_target)
    q_start = pin.neutral(pin_model)
    q_target = np.array([0.7, -1.2, 1.5, -0.8, 1.5, 0.5]) # Dowolna konfiguracja celu

    # 2. INICJALIZACJA PYBULLET (Świat fizyczny)
    print("Otwieranie okna PyBullet...")
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.81) # Grawitacja w symulatorze
    
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=135, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.5])

    robot_id = p.loadURDF(urdf_path, useFixedBase=True)

    # Pobieramy indeksy aktywnych przegubów
    active_pb_joints = []
    for j in range(p.getNumJoints(robot_id)):
        if p.getJointInfo(robot_id, j)[2] != p.JOINT_FIXED:
            active_pb_joints.append(j)
            
    # --- !!! BARDZO WAŻNY KROK DLA STEROWANIA MOMENTOWEGO !!! ---
    # PyBullet domyślnie włącza silniki pozycyjno-prędkościowe na każdym stawie.
    # Aby sterować czystym momentem (siłą), musimy te silniki JAWNIE WYŁĄCZYĆ,
    # ustawiając ich maksymalną siłę na 0. W przeciwnym razie będą z nami walczyć!
    for j in active_pb_joints:
        p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, force=0)

    # 3. NASTAWY REGULATORA PD
    # UR10 to ciężki robot, dlatego stawy u nasady (ramię, łokieć) potrzebują dużego Kp,
    # a lekkie stawy nadgarstka znacznie mniejszego.
    kp = np.array([400.0, 800.0, 400.0, 100.0, 40.0, 10.0])
    kd = np.array([40.0,  80.0,  40.0,  10.0,  4.0,  1.0])
    
    dt = 1.0 / 240.0 # Standardowy krok czasowy PyBullet
    p.setTimeStep(dt)

    print("Regulator PD + Grawitacja uruchomiony. Naciśnij Ctrl+C, aby wyjść.")
    
    try:
        while True:
            # KROK A: Odczyt AKTUALNEJ pozycji i prędkości bezpośrednio z fizyki PyBullet
            q_real = np.zeros(pin_model.nq)
            v_real = np.zeros(pin_model.nv)
            
            for i, pb_j in enumerate(active_pb_joints):
                pos, vel, _, _ = p.getJointState(robot_id, pb_j)
                q_real[i] = pos
                v_real[i] = vel
                
            # KROK B: Obliczenie siły grawitacji działającej na robota w jego BIEŻĄCEJ pozycji q_real
            g = pin.computeGeneralizedGravity(pin_model, pin_data, q_real)
            
            # KROK C: Obliczenie uchybu (błędu) pozycji oraz prędkości
            err = q_target - q_real
            derr = 0.0 - v_real # Celujemy w zatrzymanie robota, więc docelowa prędkość = 0
            
            # KROK D: Wyznaczenie sygnału z regulatora PD
            u = kp * err + kd * derr
            
            # KROK E: Ostateczny moment siły: PD (ruch do celu) + Grawitacja (utrzymanie masy)
            tau = u + g
            
            # KROK F: Zaaplikowanie obliczonych momentów sił do silników PyBullet
            for i, pb_j in enumerate(active_pb_joints):
                p.setJointMotorControl2(robot_id, pb_j, p.TORQUE_CONTROL, force=tau[i])
                
            # Wykonanie kroku fizyki w symulatorze
            p.stepSimulation()
            time.sleep(dt)

    except KeyboardInterrupt:
        p.disconnect()
        print("\nSymulacja zakończona.")

if __name__ == '__main__':
    main()