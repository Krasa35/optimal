import time
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

# We use example_robot_data to get a perfect UR10 model out-of-the-box
import example_robot_data

def main():
    print("Loading UR10 robot model...")
    # 1. Load the robot (this includes the URDF, meshes, and kinematics)
    robot = example_robot_data.load('ur10')

    # 2. Initialize the Meshcat Visualizer
    # We pass the kinematic model, collision geometries, and visual geometries
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)

    # Start the Meshcat server and open it in the default web browser
    print("Starting Meshcat server. Look for a new tab in your browser!")
    viz.initViewer(open=True)
    
    # ADD THIS: Wait for 2 seconds before loading models
    time.sleep(2.0) 
    
    # Load the 3D meshes into the viewer
    viz.loadViewerModel()

    # 3. Setup the animation loop
    q0 = robot.q0  # Get the neutral/default joint configuration
    dt = 0.01      # 100 Hz update rate
    t = 0.0

    print("Starting animation loop. Press Ctrl+C in the terminal to stop.")
    
    try:
        while True:
            # Create a copy of the default position
            q = q0.copy()
            
            # Create a smooth movement using sine and cosine waves
            # UR10 has 6 joints. Let's move the first three (pan, lift, elbow)
            q[0] += np.sin(t) * 1.0        # Shoulder Pan
            q[1] += np.sin(t * 0.5) * 0.5  # Shoulder Lift
            q[2] += np.cos(t * 1.2) * 1.2  # Elbow
            
            # Send the new joint configuration to the browser
            viz.display(q)
            
            # Advance time
            t += dt
            time.sleep(dt)
            
    except KeyboardInterrupt:
        print("\nAnimation stopped by user.")

if __name__ == '__main__':
    main()