import numpy as np
import matplotlib.pyplot as plt
import random


def find_time_hit(v_z, dz):
    return np.roots([0.5*9.81, -v_z, dz])

def calculate_vel_vec(A, B, v_z):
    x_a, y_a, z_a = A
    x_b, y_b, z_b = B
    
    dx = x_b-x_a
    dy = y_b-y_a
    dz = z_b-z_a

    t_hit_list = find_time_hit(v_z, dz)
    real_numbers = [number for number in t_hit_list if isinstance(number, (int, float))]

    if real_numbers:
        t_hit = max(real_numbers)
    else:
        return [0,0,0]

    v_x = dx/t_hit
    v_y = dy/t_hit

    return [v_x, v_y, v_z]


def plot_trajectory(B, vel_and_A_vecs, t_max):
    g = 9.81  # acceleration due to gravity in m/s^2
    xb, yb, zb = B
    
     # Time points
    t_points = np.linspace(0, t_max, 500)

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    i = 0
    for A_vel_combo in vel_and_A_vecs:
        i+=1
        x0, y0, z0 = A_vel_combo[0]
        # Plot point A
        ax.scatter(x0, y0, z0, c='g', marker='o')

        vx, vy, vz = A_vel_combo[1]

        # Equations of motion
        x = x0 + vx * t_points
        y = y0 + vy * t_points
        z = z0 + vz * t_points - 0.5 * g * t_points ** 2

        # Plot trajectory
        ax.plot(x, y, z)



    # Plot point B
    ax.scatter(xb, yb, zb, c='r', marker='o', label='Target B')
    
    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()


A = [0, 0, 0]
B = [2, 3, 2]



#The throwing space is the points filling the box created from base_point with length len_x/y/z in the respective directions, with resolution A_resolution
#A_resolution is given in points/m
def create_points(base_point, len_x, len_y, len_z, A_resolution):
    return [
        [
            base_point[0] + x / A_resolution, 
            base_point[1] + y / A_resolution, 
            base_point[2] + z / A_resolution
        ]
        for x in range(int(len_x * A_resolution))
        for y in range(int(len_y * A_resolution))
        for z in range(int(len_z * A_resolution))
    ]


def calculate_vel_vecs_in_area(base_point, len_x, len_y, len_z, A_resolution, B, v_min, v_max, v_resolution):
    #Create points A
    A_vec = create_points(base_point,len_x, len_y, len_z, A_resolution)

    vel_A_combo = []
    for A in A_vec:
        for v_z in range (v_min, v_max*v_resolution):
            vel_vec = calculate_vel_vec(A, B, v_min + v_z/v_resolution)
            if vel_vec == [0,0,0]:
                continue
            vel_A_combo.append([A, vel_vec])

    return vel_A_combo



#Testing
combo = calculate_vel_vecs_in_area([-0.2,0.1,1.2], 0.4, 0.4, 0.4, 10, [2,0,1], 1, 10, 10)
#print(combo)
print(len(combo))

random_trajs = []
for i in range(10):
    random_trajs.append(combo[random.randint(10,len(combo))])
print(random_trajs[0], "\n", random_trajs[1], "\n", random_trajs[2])
plot_trajectory([2,0,1], random_trajs, 2)

#print(vel_and_A_vecs)
#print(len(vel_and_A_vecs))

#plot_trajectory(B, vel_and_A_vecs, 3)





