"""
configuration class for the MultiUavsEnv2D class. It contains the
parameters that will be used as default
"""

# global constant are ok, global variable no
MULTI_UAV_ENV_2D_CONFIG = {
    "x": 10,
    "y": 10,
    "obs_x": 2,
    "obs_y": 1,
    "n_actions": 5,
    "n_agents": 4,
    "actions": [
        (0, 0),  # stay
        (0, 1),  # up
        (0, -1),  # down
        (-1, 0),  # left
        (1, 0)  # right
    ],
    "id_to_actions": {
        "stay": 0,
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4

    }
}
