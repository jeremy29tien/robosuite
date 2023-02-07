import numpy as np

"""
A collection of helper methods that implement features (to be used later down the line for comparisons)
for the Lift task.
"""


def speed(gym_obs):
    object_state = gym_obs[0:10]
    proprio_state = gym_obs[10:50]
    joint_vel = proprio_state[14:21]
    gripper_qvel = proprio_state[34:40]
    return np.linalg.norm(gripper_qvel, 2)  # Returning L2 norm of the gripper q-velocities as the speed. 


def height(gym_obs):
    object_state = gym_obs[0:10]
    proprio_state = gym_obs[10:50]
    eef_pos = proprio_state[21:24]
    return eef_pos[2]  # Returning z-component of end-effector position as height.


def distance_to_bottle(gym_obs):
    pass
