import numpy as np
from robosuite.environments.manipulation.lift_features import speed, height, distance_to_bottle


# This function takes in two trajectories in the form of LISTS of (observation, action) pairs.
def generate_synthetic_comparisons(traj1, traj2, feature_name):
    horizon = len(traj1)
    traj1_feature_values = None
    traj2_feature_values = None
    if feature_name == "speed":
        traj1_feature_values = [speed(traj1[t]) for t in range(horizon)]
        traj2_feature_values = [speed(traj2[t]) for t in range(horizon)]

        if np.mean(traj1_feature_values) > np.mean(traj2_feature_values):
            return ["The first trajectory is faster than the second trajectory.",
                    "The second trajectory is slower than the first trajectory."
                    ]
        else:
            return ["The first trajectory is slower than the second trajectory.",
                    "The second trajectory is faster than the first trajectory."
                    ]

    elif feature_name == "height":
        traj1_feature_values = [height(traj1[t]) for t in range(horizon)]
        traj2_feature_values = [height(traj2[t]) for t in range(horizon)]

        if np.mean(traj1_feature_values) > np.mean(traj2_feature_values):
            return ["The first trajectory is higher than the second trajectory.",
                    "The second trajectory is lower than the first trajectory."
                    ]
        else:
            return ["The first trajectory is lower than the second trajectory.",
                    "The second trajectory is higher than the first trajectory."
                    ]

    elif feature_name == "distance_to_bottle":
        traj1_feature_values = [distance_to_bottle(traj1[t]) for t in range(horizon)]
        traj2_feature_values = [distance_to_bottle(traj2[t]) for t in range(horizon)]

        if np.mean(traj1_feature_values) > np.mean(traj2_feature_values):
            return ["The first trajectory is further from the bottle than the second trajectory.",
                    "The second trajectory is closer to the bottle than the first trajectory."
                    ]
        else:
            return ["The first trajectory is closer to the bottle than the second trajectory.",
                    "The second trajectory is further from the bottle than the first trajectory."
                    ]

