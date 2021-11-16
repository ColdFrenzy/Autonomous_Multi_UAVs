import sys
import os
sys.path.insert(1, os.path.abspath(os.path.curdir))
from gym_environment.multi_uav_env_2d import MultiUavsEnv2D
import tensorflow as tf
import random
import time
import numpy as np


if __name__ == "__main__":
    env = MultiUavsEnv2D(n_agents=4)
    # uavs_initial_pos = np.array([[0, 0], [3, 5], [7, 9], [8, 6]])
    # obstacles_pos = np.array([[5, 5], [7, 2], [8, 8], [3, 4]])

    uavs_initial_pos, targets_pos, obstacles_pos = env.random_initialization()

    obs = env.reset(uavs_initial_pos, targets_pos, obstacles_pos)
    count = 0
    action_dict = {}
    while count <= 1000:
        if env.done["__all__"] is True:
            break
        for agent in env.done:
            if agent == "__all__":
                continue
            if env.done[agent] is False:
                action_dict[agent] = random.randint(0, 4)
        # stay 0 (0,0), up 1 (-1,0), down 2 (1,0), left 3 (0, -1),
        # right 4 (0, 1)
        obs, reward, done, info = env.step(action_dict)
        env.render()
        time.sleep(3)

        count += 1

    env.close()
