import pybullet as p
import pybullet_data
import time
import numpy as np
import pinocchio as pin

def main():
    # 1. Inicjalizacja Pinocchio
    urdf_path = "../models/ur10/ur10.urdf" 
    print("Wczytywanie modelu do Pinocchio...")
    pin_model = pin.buildModelFromUrdf(urdf_path)
    pin_data = pin_model.createData()
    
    # Startujemy z pozycji neutralnej (wyprostowany pionowo lub domyślny URDF)
    q = pin.neutral(pin_model)

    # 2. Bezpieczne znalezienie ID efektora końcowego (wrist_3_link lub tool0)
    ee_frame_name = "wrist_3_link"
    if not pin_model.existFrame(ee_frame_name):
        if pin_model.existFrame("tool0"):
            ee_frame_name = "tool0"
        else:
            ee_frame_name = pin_model.frames[-1].name
            
    ee_frame_id = pin_model.getFrameId(ee_frame_name)
    print(f"Znaleziono ramkę efektora: {ee_frame_name} (ID: {ee_frame_id})")

    # Obliczamy początkową pozycję w przestrzeni XYZ, aby wokół niej kręcić okrąg
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    initial_ee_pos = pin_data.oMf[ee_frame_id].translation.copy()
    print(f"Początkowa pozycja XYZ efektora: {initial_ee_pos}")

    # 3. Inicjalizacja PyBullet (Okno graficzne)
    print("Otwieranie okna PyBullet...")
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=135, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.8])

    robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.8], baseOrientation=[0, 0, 0, 1], useFixedBase=True)

    # Mapowanie aktywnych przegubów między PyBullet a Pinocchio
    active_pb_joints = []
    for j in range(p.getNumJoints(robot_id)):
        if p.getJointInfo(robot_id, j)[2] != p.JOINT_FIXED:
            active_pb_joints.append(j)

    # 4. Pętla ruchu w przestrzeni XYZ
    t = 0.0
    dt = 0.01  # Krok czasowy (100 Hz)
    gain = 5.0  # Współczynnik wzmocnienia IK (szybkość zbiegania do celu)
    damp = 1e-4  # Tłumienie osobliwości matematycznych

    print("Uruchomiono rysowanie okręgu w XYZ! Naciśnij Ctrl+C, aby wyjść.")
    
    try:
        while True:
            # KROK A: Przeliczenie kinematyki w Pinocchio dla aktualnego q
            pin.forwardKinematics(pin_model, pin_data, q)
            pin.updateFramePlacements(pin_model, pin_data)
            
            # Pobranie aktualnej pozycji efektora
            current_ee_pos = pin_data.oMf[ee_frame_id].translation
            
            # KROK B: Zdefiniowanie celu w XYZ (Trajektoria kołowa w płaszczyźnie Y-Z)
            # Promień okręgu r = 0.15 metra
            target_ee_pos = initial_ee_pos.copy()
            target_ee_pos[1] += 0.15 * np.sin(t)  # Ruch w osi Y
            target_ee_pos[2] += 0.15 * (np.cos(t) - 1.0)  # Ruch w osi Z

            # KROK C: Obliczenie błędu pozycji kartezjańskiej
            err_xyz = target_ee_pos - current_ee_pos
            
            # KROK D: Pobranie Jakobianu efektora
            # LOCAL_WORLD_ALIGNED oznacza, że osie są zorientowane tak jak układ świata, co ułatwia sterowanie XYZ
            J = pin.computeFrameJacobian(pin_model, pin_data, q, ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J_xyz = J[:3, :]  # Interesują nas tylko wiersze odpowiedzialne za pozycję liniową (X, Y, Z)
            
            # KROK E: Przeliczenie błędu XYZ na prędkości przegubów (v = J_pseudo_inverse * err)
            # Używamy algorytmu Damped Least Squares (solve z regularyzacją), aby uniknąć dzielenia przez zero w trudnych układach ramienia
            v_joints = J_xyz.T @ np.linalg.solve(J_xyz @ J_xyz.T + damp * np.eye(3), err_xyz * gain)
            
            # KROK F: Integracja prędkości – wyznaczenie nowego q
            q = pin.integrate(pin_model, q, v_joints * dt)
            
            # KROK G: Aktualizacja wizualizacji w PyBullet
            for i, pb_j in enumerate(active_pb_joints):
                p.resetJointState(robot_id, pb_j, q[i])
                
            # Opcjonalne: Rysowanie czerwonej kropki w miejscu celu dla ułatwienia debugowania
            p.addUserDebugLine(target_ee_pos, target_ee_pos + np.array([0,0,0.01]), [1, 0, 0], lifeTime=1)

            p.stepSimulation()
            time.sleep(dt)
            t += dt

    except KeyboardInterrupt:
        p.disconnect()
        print("\nWizualizacja zakończona.")

if __name__ == '__main__':
    main()