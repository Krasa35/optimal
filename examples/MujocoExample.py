from utils.MujocoController import MujocoController

# create controller instance
controller = MujocoController()

# Display robot information
controller.show_model_info()

# Pick & Place with pauses and plots
controller.move_group_to_joint_target("Arm", [0.0, -1.56, 1.56, -1.56, -1.56])
controller.stay(100)
controller.move_ee([0.0, -0.6, 1.1])
controller.stay(100)
controller.move_ee([0.0, -0.6, 0.98])
controller.stay(100)
controller.close_gripper(max_steps=300)
controller.move_ee([0.0, -0.6, 1.1])
controller.stay(100)

controller.clear_plot_lists()
controller.stay(1000)
controller.create_joint_angle_plot()
controller.create_torque_plot()

controller.move_group_to_joint_target("Arm", [0.0, -1.56, 1.56, -1.56, -1.56])
controller.stay(1000)
controller.open_gripper()
controller.stay(1000)
controller.move_ee([0.0, -0.6, 1.1])
controller.stay(100)
# controller.create_joint_angle_plot()
# controller.create_torque_plot()
input("waiting....")