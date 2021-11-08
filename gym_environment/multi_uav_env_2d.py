import os
import numpy as np
from gym.spaces import Box, Dict, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from config.multi_uav_env_2d_config import MULTI_UAV_ENV_2D_CONFIG as conf_1
img_dir = os.path.join(os.curdir, "images")
uav_img = os.path.join(img_dir, "UAV_Icon.png")


class MultiUavsEnv2D(MultiAgentEnv):
    """MultiUavsEnv2D class.

    An environment that hosts multiple independent agents. Agents are
    identified by (string) agent ids. Note that these "agents" here
    are not to be confused with RLlib agents.
    The environment is represented as a 2D grid where the
    """

    def __init__(self, x=conf_1["x"],
                 y=conf_1["y"], n_actions=conf_1["n_actions"],
                 n_agents=conf_1["n_agents"],
                 obs_x=conf_1["obs_x"],
                 obs_y=conf_1["obs_y"],
                 actions=conf_1["actions"]):
        """__init__ method.

        Parameters
        ----------
        x : int
            Number of row of the 2D grid
        y : int
            Number of column of the 2D grid
        n_actions: int
            Number of actions an agent can perform
        """
        self.x = x
        self.y = y
        self.grid_size = (x, y)
        self.obs_x = obs_x
        self.obs_y = obs_y
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.actions = actions
        self.viewer = None
        self.observation_space = Dict(
            {
                "uav"+str(_)+"obs": Box(low=-1, high=1, shape=(obs_x, obs_y),
                                        dtype=np.float32) for _ in
                range(self.n_agents)
            }
        )
        self.action_space = Discrete(n_actions)

        self.all_agents_pos = {
            "uav_"+str(_): None for _ in range(self.n_agents)}

    def reset(self, initial_pos=None):
        """reset method.

        If an initial position is not specified, the agents will be allocated
        randomly
        """
        self.grid = np.zeros(self.grid_size, dtype=np.float32)
        self.done = {
            "uav_"+str(_): False for _ in range(self.n_agents)}
        self.done["__all__"] = False
        if initial_pos is None:
            pass
        else:
            # insert agents in the map
            for n in range(self.n_agents):
                i, j = initial_pos[n]
                self.all_agents_pos["uav_"+str(n)] = np.array([i, j])
                self.grid[i, j] = 1
        return self.get_all_agents_observation()

    def step(self, action_dict):
        """step method.

        Update the grid by using the actions of the agents. An agent collides
        with another agent only if their final positions coincide. In such
        case, both the agent are removed.

        Returns
        -------
        Tuple containing 1) new observations for
        each ready agent, 2) reward values for each ready agent.
        3) Done values for each ready agent. The special key
        "__all__" (required) is used to indicate env termination.
        4) Optional info values for each agent id.
        """
        # remove all the agents from the grid
        self.grid[self.grid == 1] = 0
        # at the moment reward is always 0 and -1 when collision happen
        obs, reward, done, info = {}, {}, {}, {}
        new_positions = []
        indx_to_remove = []
        for agent in action_dict:
            action = action_dict[agent]
            agent_pos_x, agent_pos_y = self.all_agents_pos[agent]
            agent_new_pos_x = agent_pos_x + action[0]
            agent_new_pos_y = agent_pos_y + action[1]
            # check if the new position is on grid
            if self.is_on_grid(agent_new_pos_x, agent_new_pos_y):
                # check if the new position collides with an obstacle
                if self.grid[agent_new_pos_x, agent_new_pos_y] == 0:
                    # check if there is already another agent in that position
                    if [agent_new_pos_x, agent_new_pos_y] in new_positions:
                        indx = new_positions.index([agent_new_pos_x,
                                                    agent_new_pos_y])

                        if indx not in indx_to_remove:
                            indx_to_remove.append(indx)
                        indx_to_remove.append(len(new_positions))
                    new_positions.append([agent_new_pos_x, agent_new_pos_y])

                else:
                    indx_to_remove.append(len(new_positions))
                    new_positions.append([agent_new_pos_x, agent_new_pos_y])
            else:
                indx_to_remove.append(len(new_positions))
                new_positions.append([agent_new_pos_x, agent_new_pos_y])
        print(new_positions)
        # update position of the agents
        for i, agent in enumerate(action_dict):
            if i in indx_to_remove:
                self.all_agents_pos[agent] = np.array([-1, -1])
                self.done[agent] = True
                done[agent] = True
                reward[agent] = -1.0
            else:
                self.all_agents_pos[agent] = np.array(new_positions[i])
                self.grid[new_positions[i][0], new_positions[i][1]] = 1
                done[agent] = False
                reward[agent] = 0.0

        # after all the position have been updated, we can retrieve agents
        # observations
        for agent in done:
            if agent == "__all__":
                continue
            else:
                obs[agent] = self.get_agent_observation(agent)

        return obs, reward, done, info

    def get_all_agents_observation(self):
        """get_all_agents_observation method.

        return the observations for all the agents that are still in the
        environment

        """
        all_agents_obs = {"uav_"+str(_): None for _ in range(self.n_agents)}
        for agent_id in self.all_agents_pos:
            all_agents_obs[agent_id] = self.get_agent_observation(agent_id)

        return all_agents_obs

    def get_agent_observation(self, agent_id):
        """get_agent_observation method.

        Return the observation of the agent with id = agent_id
        Parameters
        ----------
        agent_id: str
        """
        i, j = self.all_agents_pos[agent_id]
        obs = np.full((self.obs_x*2+1, self.obs_y*2+1), -1, dtype=np.float32)
        for x in range(self.obs_x+1):
            for y in range(self.obs_y+1):
                if x == 0 and y == 0:
                    # agent location
                    obs[self.obs_x, self.obs_y] = 1
                    continue

                # redundant conditional checking when x = 0 and y = 0
                if self.is_on_grid(i-x, j-y):
                    obs[self.obs_x-x, self.obs_y-y] = self.grid[i-x, j-y]
                if self.is_on_grid(i-x, j+y):
                    obs[self.obs_x-x, self.obs_y+y] = self.grid[i-x, j+y]
                if self.is_on_grid(i+x, j-y):
                    obs[self.obs_x+x, self.obs_y-y] = self.grid[i+x, j-y]
                if self.is_on_grid(i+x, j+y):
                    obs[self.obs_x+x, self.obs_y+y] = self.grid[i+x, j+y]

        return obs

    def is_on_grid(self, x, y):
        """is_on_grid function.

        Check if the position (x, y) is on the grid
        """
        return x >= 0 and x < self.x and y >= 0 and y < self.y

    def is_valid_action(self, x, y, act):
        """is_valid_action function.

        Check if the action is valid, i.e. it leads to a valid state.
        the function requires the position of the agent (x,y) and the id of the
        action picked
        """
        x = x + act[0]
        y = y + act[1]
        # if the new position is on grid, check if it collide with an obstacle
        if self.is_on_grid(x, y):
            if self.grid(x, y) == 0:
                return

    def render(self, mode="human", screen_width=600, screen_height=400):
        """render method.

        Render the 2D scenario
        """
        if mode == "human":
            # new elements are added in the bottom left corner. To center them,
            # translate by (screen_width/2, screen_height/2)
            from gym.envs.classic_control import rendering
            cell_size = 50
            screen_width = cell_size * self.x
            screen_height = cell_size * self.y
            background_width = screen_width
            background_height = screen_height
            # this is necessary otherwise it crates new
            if self.viewer is None:
                # if viewer is None, let's initialize the grid
                self.viewer = rendering.Viewer(screen_width, screen_height)
                l, r, t, b = (
                    -background_width / 2,
                    background_width / 2,
                    background_height / 2,
                    -background_height / 2,
                )
                background = rendering.FilledPolygon(
                    [(l, b), (l, t), (r, t), (r, b)])
                backtrans = rendering.Transform(
                    translation=(screen_width / 2, screen_height / 2)
                )
                background.add_attr(backtrans)
                background.set_color(1.0, 1.0, 1.0)
                self.viewer.add_geom(background)
                for i in range(self.x):
                    row_line = rendering.Line(start=(0, i*cell_size),
                                              end=(screen_width, i*cell_size))
                    self.viewer.add_geom(row_line)
                for j in range(self.y):
                    col_line = rendering.Line(start=(j*cell_size, 0),
                                              end=(j*cell_size, screen_height))
                    self.viewer.add_geom(col_line)

                # self.uavs = {}
                for agent in self.all_agents_pos:
                    agent_pos_x, agent_pos_y = self.all_agents_pos[agent]
                    if [agent_pos_x, agent_pos_y] == [-1, -1]:
                        continue
                    uav = rendering.Image(uav_img, cell_size, cell_size)
                    uav_transf = rendering.Transform(
                        translation=(
                            agent_pos_x*cell_size + cell_size/2,
                            agent_pos_y*cell_size + cell_size/2
                        )
                    )
                    uav.add_attr(uav_transf)
                    self.viewer.add_onetime(uav)
                    # self.uavs[agent] = uav
            else:
                for agent in self.all_agents_pos:
                    agent_pos_x, agent_pos_y = self.all_agents_pos[agent]
                    if [agent_pos_x, agent_pos_y] == [-1, -1]:
                        continue
                    uav = rendering.Image(uav_img, cell_size, cell_size)
                    uav_transf = rendering.Transform(
                        translation=(
                            agent_pos_x*cell_size + cell_size/2,
                            agent_pos_y*cell_size + cell_size/2
                        )
                    )
                    uav.add_attr(uav_transf)
                    self.viewer.add_onetime(uav)

            # for uav in self.uavs:
            #     uav.add_attr()

            return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        """close method.

        Close the rendering view
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    # random_input = tf.ones((20, 10))
    # test_model = tf.keras.layers.Dense(100)
    # output = test_model(random_input)
    # print("Hello, the output is " + str(output))
    env = MultiUavsEnv2D(n_agents=4)
    initial_pos = np.array([[0, 0], [3, 5], [20, 10], [31, 15]])
    obs = env.reset(initial_pos)
    grid_0 = env.grid
    # stay, up, down, left, right
    action_dict = {"uav_0": 2, "uav_1": 3, "uav_2": 0, "uav_3": 4}
    obs, reward, done, info = env.step(action_dict)
    grid_1 = env.grid
