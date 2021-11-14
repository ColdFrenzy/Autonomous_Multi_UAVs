"""multi_uav_env_2d_config file.

configuration class for the MultiUavsEnv2D class. It contains the
parameters that will be used as default to generate the environment.
"""

# global constant are ok, global variable no
MULTI_UAV_ENV_2D_CONFIG = {
    # row
    "x": 10,
    # col
    "y": 20,
    "obs_x": 1,
    "obs_y": 1,
    "n_actions": 5,
    "n_agents": 4,
    "n_obstacles": 20,
    "n_targets": 4,
    "task": ["navigation"],
    # cell_size is used by the rendering function
    "cell_size": 50,
    "actions": [
        (0, 0),  # stay 0
        (-1, 0),  # up 1
        (1, 0),  # down 2
        (0, -1),  # left 3
        (0, 1)  # right 4
    ],
    "id_to_actions": {
        "stay": 0,
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4

    },
    "max_charge": 3.0,
    # energy consumption associated with an action.
    # if the uav keep moving in the same direction it will consume less energy
    # if it stay still or it accellerate to change direction, it consumes a
    # lot of energy
    "energy_consumption": {
        "still": 0.5,
        "same": 0.2,
        "different": 0.4
    }
}
