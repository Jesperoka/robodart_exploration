import matplotlib.pyplot as plt
import numpy as np
from movement_primitives.dmp import DMPWithFinalVelocity
from json import load

import panda_py
from panda_py import controllers, libfranka
from plot_dmp import plot_dmp

SHOP_FLOOR_IP = "10.0.0.2"  # hostname for the workshop floor, i.e. the Franka Emika Desk
ROBOT_IP = "192.168.0.1"    # don't know if we ever need robot IP

if __name__ == '__main__':

    panda = panda_py.Panda(SHOP_FLOOR_IP)

    with open("trajectory_pos.txt") as f:
        traj = load(f)

    # poses = [
    #     [1.6715230781752977, 0.7786007218522945, -0.008575725651101039, -1.3165846017619227, 1.6955766522261893, 1.6617156298160554, 1.0561511186108676],
    #     [1.770186800994371, -0.195214006347574, -0.03874896530414883, -2.4387160534825463, 1.7346712555814974, 1.6571671055952706, 0.5485339650304526],
    #     [1.7150186220361114, -1.3855727591550477, 0.08130718343090593, -2.863084024128161, 1.6839168995780973, 1.7103670847051442, 0.3588397114804476],
    #     [1.7220136104215857, -1.7274021764393894, 0.10320540175662729, -2.6192889615219253, 1.6862640565501317, 1.6820221564873732, -0.022574487214055815],
    #     [1.7104714936779364, -1.6578512582109683, 0.11365917445857268, -1.9442497780573995, 1.5803783130015552, 1.7333044178485868, -1.093402174974143],
    #     [1.6339732447046078, -1.7479344192806041, 0.17568994928870282, -1.2690061736525151, 1.5489794883992938, 1.7441416631836506, -2.470402759524658]
    # ]


    poses = [
        [1.6670570549906703, 0.7845368315797371, 0.03746998061840994, -1.5749109789782434, 1.625546125738416, 1.6306104865868887, 0.6355820243902918],
        [1.667575411668995, 0.4241930532664691, 0.01912996098744228, -1.8464228583983358, 1.6072097168233659, 1.6316357373396553, 0.34412831266081634],
        [1.6554974093750903, 0.12334005988818839, 0.010624840188100039, -1.9409026279784083, 1.5669333261118994, 1.6316442743142443, 0.18638848995487628],
        [1.6426656754372413, -0.6178403747256765, 0.014696339967839176, -2.2897818158802234, 1.5606716165807484, 1.6001181999047598, -0.04526729967693488],
        [1.6434303555133047, -1.016679049744195, 0.012871403555596862, -2.1535226010439708, 1.560899600373374, 1.5995299274921415, -0.34409441173646826],
        [1.6434826836230463, -1.318603079611795, -0.0031690987100857393, -2.0276529012832336, 1.5557613700495907, 1.5819362745688021, -0.6903655359066461],
    ]



    # Variables for np.linspace arguments
    num_samples = 1000  # Number of samples for interpolation
    axis = 0  # Axis along which to interpolate

    # Initialize an empty list to store interpolated arrays
    interpolated_poses = []

    # Loop through each pair of consecutive poses and interpolate
    for i in range(len(poses) - 1):
        start_pose = np.array(poses[i])
        end_pose = np.array(poses[i + 1])
        interpolated = np.linspace(start_pose, end_pose, num_samples, axis=axis)
        interpolated_poses.append(interpolated)

    # Concatenate all interpolated arrays
    concatenated_interpolated_poses = np.concatenate(interpolated_poses, axis=axis)

    # Demonstrated trajectory
    # demo_y = np.array(traj)
    demo_y = concatenated_interpolated_poses

    # DMP hyperparameters
    N = max(demo_y.shape)
    execution_time = 10
    dt = execution_time / N
    n_weights_per_dim = 20
    T = np.linspace(0, execution_time, N)   

    # Initialize DMP
    dmp = DMPWithFinalVelocity(n_dims=7, execution_time=execution_time, dt=dt, n_weights_per_dim=n_weights_per_dim)
    dmp.imitate(T, demo_y)
    
    # Set target endpoint of trajectory
    dmp.configure(goal_y=np.array(demo_y[-1,:]), goal_yd=np.array([0, 0, 0, 0.3, 0, 0, -1]))

    dmp_t, dmp_y = dmp.open_loop(run_t=execution_time)
    dmp_yd = (1.0/dmp.dt_)*np.gradient(dmp_y, axis=0)

    plot_dmp(execution_time, dt, demo_y, dmp_y, dmp_yd)

    panda.move_to_joint_position(dmp_y[0])
    input("Press any key to start")

    # Follow trajectory
    i = 0
    frequency = 1000
    ctrl = controllers.JointPosition()
    panda.start_controller(ctrl)
    with panda.create_context(frequency=frequency, max_runtime=execution_time) as ctx:
        panda.enable_logging(frequency*execution_time)
        while ctx.ok() and i < len(dmp_y):
            ctrl.set_control(dmp_y[i], dmp_yd[i])
            i += 1

    # Extract results
    result_y = np.array(panda.get_log()['q'])
    result_yd = np.array(panda.get_log()['dq'])

    plot_dmp(execution_time, dt, demo_y, dmp_y, dmp_yd, result_y, result_yd)