import numpy as np

LOWER_JOINT_LIMITS = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
UPPER_JOINT_LIMITS = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
JOINT_VELOCITY_LIMITS = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])

def limit_position(q):
    too_high = np.greater(q, UPPER_JOINT_LIMITS)
    too_low = np.greater(LOWER_JOINT_LIMITS, q)

    for i, val in enumerate(too_high):
        if val:
            q[i] = UPPER_JOINT_LIMITS[i]-0.01
    
    for i, val in enumerate(too_low):
        if val:
            q[i] = LOWER_JOINT_LIMITS[i]+0.01

    return q


def limit_velocity(q):
    too_high = np.greater(np.abs(q), JOINT_VELOCITY_LIMITS)

    for i, val in enumerate(too_high):
        if val:
            q[i] = JOINT_VELOCITY_LIMITS[i]-0.01

    return q