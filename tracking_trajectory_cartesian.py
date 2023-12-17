import matplotlib.pyplot as plt
import numpy as np
from movement_primitives.dmp import DMPWithFinalVelocity
from json import load
from plot_dmp import plot_dmp
from convert_pose_expression import cartesian_pose_to_transformation_matrix, transformation_matrix_to_cartesian_pose, quaternion_to_rotation_vector, rotation_vector_to_quaternion
from scipy.spatial.transform import Rotation as R
from target_to_velocity_map import calculate_launch_point_and_velocity_vectors
from jacobian import get_jacobian, get_jacobian_3_joints
from limits import limit_joint_position, limit_joint_velocity, limit_cartesian_velocity
from create_interpolation import create_interpolation
import panda_py
from panda_py import controllers, libfranka

SHOP_FLOOR_IP = "10.0.0.2"  # hostname for the workshop floor, i.e. the Franka Emika Desk
ROBOT_IP = "192.168.0.1"    # don't know if we ever need robot IP

if __name__ == '__main__':

    # panda = panda_py.Panda(SHOP_FLOOR_IP)

    ####################################################

    # Interpolating trajectory from set of poses
    # Cartesian poses provided

    # poses = [
    #     [-0.26684645, 0.72326601, 0.29045006, -0.33319444, -0.63154138, 0.2573719, 0.65107346],
    #     [-0.27151081, 0.59114054, 0.43162935, -0.2741086, -0.65613221, 0.21893884, 0.66814728],
    #     [-0.26576335, 0.39897345, 0.58287283, -0.1481909, -0.67848496, 0.13447379, 0.70683408],
    #     [-0.25124114, 0.21696712, 0.7275276, 0.05008003, -0.69233572, -0.0412439, 0.71865304],
    #     [-0.24030373, 0.0226624, 0.8485015, 0.36221429, -0.58070067, -0.34858058, 0.6403742 ],
    #     [-0.26317091, -0.19931589, 0.91034192, 0.61383395, -0.34359184, -0.57310566, 0.42035989]
    # ]

    joint_poses = np.array([
        [1.6670570549906703, 0.7845368315797371, 0.03746998061840994, -1.5749109789782434, 1.625546125738416, 1.6306104865868887, 0.6355820243902918],
        [1.667575411668995, 0.4241930532664691, 0.01912996098744228, -1.8464228583983358, 1.6072097168233659, 1.6316357373396553, 0.34412831266081634],
        [1.6554974093750903, 0.12334005988818839, 0.010624840188100039, -1.9409026279784083, 1.5669333261118994, 1.6316442743142443, 0.18638848995487628],
        [1.6426656754372413, -0.6178403747256765, 0.014696339967839176, -2.2897818158802234, 1.5606716165807484, 1.6001181999047598, -0.04526729967693488],
        [1.6434303555133047, -1.016679049744195, 0.012871403555596862, -2.1535226010439708, 1.560899600373374, 1.5995299274921415, -0.34409441173646826],
        [1.6434826836230463, -1.318603079611795, -0.0031690987100857393, -2.0276529012832336, 1.5557613700495907, 1.5819362745688021, -0.6903655359066461],
    ])
    
    # Generate cartesian poses from joint poses
    poses = np.zeros_like(joint_poses)
    for i, pose in enumerate(joint_poses):
        b = transformation_matrix_to_cartesian_pose(panda_py.fk(pose))
        poses[i] = b

    # Variables for np.linspace arguments 
    num_samples = 1000  # Number of samples for interpolation
    axis = 0  # Axis along which to interpolate
    concatenated_interpolated_poses = create_interpolation(poses, num_samples, axis)

    with open("trajectory_pos.txt") as f:
        traj = np.array(load(f))

    poses2 = np.zeros_like(traj)
    for i, pose in enumerate(traj):
        b = transformation_matrix_to_cartesian_pose(panda_py.fk(pose))
        poses2[i] = b


    #demo_y = concatenated_interpolated_poses
    demo_y = poses2
    
    N = max(demo_y.shape)
    execution_time = 25
    dt = execution_time / N
    n_weights_per_dim = 15
    T = np.linspace(0, execution_time, N)   

    dmp = DMPWithFinalVelocity(n_dims=7, execution_time=execution_time, dt=dt, n_weights_per_dim=n_weights_per_dim)
    dmp.imitate(T, demo_y)


    g = 0.98219  # local gravitational acceleration in Trondheim at 45m according to WolframAlpha 
    base_pt = np.array([0.2, 0.1, 2.2])
    dim_lengths = (0.4, 0.4, 0.4)
    volume_res = 10
    target_pt = np.array([2.37, 0.0, 1.73])
    vel_limits = (4.0, 6.0)
    vel_res = 10

    launch_point_and_velocities = calculate_launch_point_and_velocity_vectors(base_pt, *dim_lengths, volume_res, target_pt, *vel_limits, vel_res, g=g)

    # Example goal position and velocity from target_to_velocity_map.py
    cartesian_goal_pos, cartesian_goal_vel = launch_point_and_velocities[500]

    # DH parameters of end effector (a = 1.34m + 0.1m = 1.44m, d = 0.11m, alpha = 0, theta = 0)
    translational_goal_vel = limit_cartesian_velocity(cartesian_goal_vel)
    diff = cartesian_goal_vel-translational_goal_vel
    angular_goal_vel = rotation_vector_to_quaternion([0, 0, np.linalg.norm(diff)/1.44])


    # 45 degrees upwards towards dart board
    cartesian_goal_orientation = np.array([0.2705981, -0.6532815, -0.2705981, 0.6532815])

    goal_y = np.append(cartesian_goal_pos, cartesian_goal_orientation)

    dmp.configure(goal_y=goal_y, goal_yd=np.array(np.append(translational_goal_vel, angular_goal_vel)))
    # dmp.configure(goal_y=goal_y)

    T, dmp_y = dmp.open_loop(run_t=execution_time)
    dmp_yd = (1.0/dmp.dt_)*np.gradient(dmp_y, axis=0)

    plot_dmp(execution_time, dt, demo_y, dmp_y, dmp_yd, filename="cartesian_dmp_result.pdf")


    # Generate trajectory in joint space for control purpose
    joint_trajectory = np.zeros_like(dmp_y)
    joint_trajectory[0] = joint_poses[0]
    
    # I do not konw what alpha should be
    alpha = 0.1
    print("alpha = ", alpha)
    epsilon = 1
    for index, pose in enumerate(dmp_y):
        if index > 0:

            q = joint_trajectory[index-1]
            J = get_jacobian(q)
            pseudo_J = np.linalg.pinv(J)
            x_desired = np.zeros(6)
            x_desired[0:3] = pose[0:3]
            x_desired[3:7] = quaternion_to_rotation_vector(pose[3:7])
            delta_x = np.zeros(6)
            delta_x[0:3] = x_desired[0:3]-dmp_y[index-1][0:3]
            delta_x[3:6] = x_desired[0:3]-quaternion_to_rotation_vector(dmp_y[index-1][3:7])

            # q_next = q_next + alpha*(pseudo_J @ delta_x)

#############################################
            q_next = q
            # print(index)
            # while np.linalg.norm(delta_x) > epsilon:
            for i in range(25):
                q_next = q_next + alpha*(pseudo_J @ delta_x)
                q_next = limit_joint_position(q_next)
                J = get_jacobian(q_next)
                pseudo_J = np.linalg.pinv(J)
                x_estimate = transformation_matrix_to_cartesian_pose(panda_py.fk(q_next))
                delta_x = x_desired-np.append(x_estimate[0:3], quaternion_to_rotation_vector(x_estimate[3:7]))
#############################################

            q_next = limit_joint_position(q_next)

            joint_trajectory[index] = q_next
            joint_trajectory[index, 0] = joint_trajectory[0,0]
            joint_trajectory[index, 2] = joint_trajectory[0,2]
            joint_trajectory[index, 4] = joint_trajectory[0,4]
            joint_trajectory[index, 5] = joint_trajectory[0,5]


    # Calculate q_dot of the end pose
    q_dot = np.linalg.pinv(get_jacobian_3_joints(joint_trajectory[-1])) @ np.append(translational_goal_vel, [0, 0, np.linalg.norm(diff)/1.44])
    q_dot = limit_joint_velocity(q_dot)

    concatenated_interpolated_poses = create_interpolation(joint_poses, num_samples, axis)

    new_traj = concatenated_interpolated_poses
    N = max(new_traj.shape)
    n_weights_per_dim = 15
    # execution_time = 25
    T = np.linspace(0, execution_time, N)   

    dmp = DMPWithFinalVelocity(n_dims=7, execution_time=execution_time, dt=dt, n_weights_per_dim=n_weights_per_dim)
    dmp.imitate(T, new_traj)

    dmp.configure(goal_y=joint_trajectory[-1], goal_yd=q_dot)
    T, joint_dmp_y = dmp.open_loop(run_t=execution_time)
    joint_dmp_yd = (1.0/dmp.dt_)*np.gradient(joint_dmp_y, axis=0)


    # Plot target trajectory and calculated DMP in joint space
    plot_dmp(execution_time, dt, new_traj, joint_dmp_y, joint_dmp_yd, filename="joint_space_dmp_result.pdf", space="joint")


    # panda.move_to_joint_position(joint_dmp_y[0])
    # input("Press any key to start")

    # i = 0
    # ctrl = controllers.JointPosition()
    # panda.start_controller(ctrl)
    # with panda.create_context(frequency=1000, max_runtime=execution_time) as ctx:
    #     while ctx.ok():
    #         ctrl.set_control(joint_dmp_y[i], joint_dmp_yd[i])
    #         i += 1


    # # Extract results
    # result_y = np.array(panda.get_log()['q'])
    # result_yd = np.array(panda.get_log()['dq'])

    # plot_dmp(execution_time, dt, new_traj, joint_dmp_y, joint_dmp_yd, result_y, result_yd, filename="result.pdf")