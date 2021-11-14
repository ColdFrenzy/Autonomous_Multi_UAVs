from gym_environment.multi_uav_env_2d import MultiUavsEnv2D
import tensorflow as tf
import numpy as np
import sys
import os
sys.path.insert(1, os.path.abspath(os.path.curdir))


if __name__ == "__main__":
    env = MultiUavsEnv2D(n_agents=4)
    # uavs_initial_pos = np.array([[0, 0], [3, 5], [7, 9], [8, 6]])
    # obstacles_pos = np.array([[5, 5], [7, 2], [8, 8], [3, 4]])

    uavs_initial_pos, targets_pos, obstacles_pos = env.random_initialization()

    obs = env.reset(uavs_initial_pos, targets_pos, obstacles_pos)
    grid_0 = np.copy(env.grid)
    for i in range(500):
        if i % 100 == 0:
            print(i)
        env.render()

    print(env.all_agents_charge)
    # stay 0 (0,0), up 1 (-1,0), down 2 (1,0), left 3 (0, -1), right 4 (0, 1)
    action_dict = {"uav_0": 1, "uav_1": 3,
                   "uav_2": 0, "uav_3": 4}
    obs, reward, done, info = env.step(action_dict)
    grid_1 = np.copy(env.grid)
    for i in range(400):
        if i % 100 == 0:
            print(i)
        env.render()

    print(env.all_agents_charge)
    action_dict = {"uav_0": 1, "uav_1": 3,
                   "uav_2": 0, "uav_3": 4}
    obs, reward, done, info = env.step(action_dict)
    for i in range(400):
        if i % 100 == 0:
            print(i)
        env.render()
    print(env.all_agents_charge)
env.close()
