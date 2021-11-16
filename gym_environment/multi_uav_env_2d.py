import os
import numpy as np
import pyglet
import random
from gym.spaces import Box, Dict, Discrete
from gym.envs.classic_control import rendering
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from config.multi_uav_env_2d_config import MULTI_UAV_ENV_2D_CONFIG as conf_1
import config.colors as colors
import config.paths as paths


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
                 n_obstacles=conf_1["n_obstacles"],
                 n_targets=conf_1["n_targets"],
                 obs_x=conf_1["obs_x"],
                 obs_y=conf_1["obs_y"],
                 actions=conf_1["actions"],
                 max_charge=conf_1["max_charge"],
                 energy_consumption=conf_1["energy_consumption"],
                 task=conf_1["task"][0],
                 cell_size=conf_1["cell_size"]):
        """__init__ method.

        initialize the environment by defining the number of agents, number of
        obstacles, grid size (x, y), max_charge, energy consumption, the task
        ecc.
        """
        self.x = x
        self.y = y
        self.grid_size = (x, y)
        self.obs_x = obs_x
        self.obs_y = obs_y
        self.max_charge = max_charge
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.n_obstacles = n_obstacles
        self.n_targets = n_targets
        self.actions = actions
        self.energy_consumption = energy_consumption
        self.task = task
        self.mode = None
        self.cell_size = cell_size
        self.viewer = None
        self.observation_space = Dict(
            {
                "uav"+str(_)+"obs": Box(low=-1, high=1, shape=(obs_x, obs_y),
                                        dtype=np.float32) for _ in
                range(self.n_agents)
            }
        )
        self.action_space = Discrete(n_actions)

        self.obstacles_pos = []
        self.targets_pos = []
        self.all_agents_charge = {
            "uav_"+str(_): self.max_charge for _ in range(self.n_agents)
        }
        self.all_agents_pos = {
            "uav_"+str(_): None for _ in range(self.n_agents)
        }
        self.all_agents_prev_act = {
            "uav_"+str(_): None for _ in range(self.n_agents)
        }
        self.uavs_traj_color = {
            "uav_"+str(_): np.random.choice(range(256), size=3)/255
            for _ in range(self.n_agents)
        }

    def random_initialization(self):
        """random_initialization method.

        Randomly initialize the agents and the obstacles over a given map
        Returns
        -------
        uavs_initial_pos: list
            list containing the initial position of the uavs
        obstacles_pos: list
            list containing the position of the obstacles
        """
        n_samples = self.n_agents + self.n_obstacles + self.n_targets
        # sample unique indexes (same thing could be done by using
        # random.shuffle and retrieving the first n elements)
        indexes = random.sample(range(self.x*self.y), n_samples)
        samples = []
        for elem in indexes:
            x = elem//self.y
            y = elem % self.y
            samples.append((x, y))
        return samples[0:self.n_agents],\
            samples[self.n_agents:self.n_agents + self.n_targets],\
            samples[self.n_agents+self.n_targets:]

    def reset(self, uavs_initial_pos=None, targets_pos=None,
              obstacles_pos=None):
        """reset method.

        If an initial position is not specified, the agents and the obstacles
        will be allocated randomly
        """
        self.grid = np.zeros(self.grid_size, dtype=np.float32)
        self.done = {
            "uav_"+str(_): False for _ in range(self.n_agents)}
        self.done["__all__"] = False
        if uavs_initial_pos is None:
            pass
        else:
            assert len(uavs_initial_pos) == self.n_agents,\
                "The specified uavs locations do not match the number of" + \
                " uavs in the environment"
            # insert agents in the map
            for n in range(self.n_agents):
                i, j = uavs_initial_pos[n]
                assert self.is_on_grid(i, j),\
                    f"The UAV position ({i, j}) is outside the grid "
                self.all_agents_pos["uav_"+str(n)] = np.array([i, j])
                self.grid[i, j] = 1
            assert len(obstacles_pos) == self.n_obstacles,\
                "The specified obstacle locations do not match the number" +\
                " of obstacles in the environment"
            for obstacle in obstacles_pos:
                i, j = obstacle
                assert self.is_on_grid(i, j),\
                    f"The obstacle position ({i, j}) is outside the grid "
                self.obstacles_pos.append(obstacle)
                self.grid[i, j] = 2
            assert len(targets_pos) == self.n_targets,\
                "The specified target positions do not match the number of" + \
                " targets in the environment"
            for target in targets_pos:
                i, j = target
                assert self.is_on_grid(i, j),\
                    f"The target position ({i, j}) is outside the grid "
                self.targets_pos.append(target)
                self.grid[i, j] = 3

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
            # assert not self.done[agent], f"{agent} is out." \
            #     " It cannot take any action "
            action_indx = action_dict[agent]
            action = self.actions[action_dict[agent]]
            agent_pos_x, agent_pos_y = self.all_agents_pos[agent]
            agent_new_pos_x = agent_pos_x + action[0]
            agent_new_pos_y = agent_pos_y + action[1]
            # update energy
            self.update_energy(agent, action_indx)
            if self.all_agents_charge[agent] <= 0:
                indx_to_remove.append(len(new_positions))
                new_positions.append([-1, -1])
                continue
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

        # update position of the agents
        for i, agent in enumerate(action_dict):
            if i in indx_to_remove:
                self.all_agents_pos[agent] = np.array([-1, -1])
                self.done[agent] = True
                self.all_agents_charge[agent] = 0
                done[agent] = True
                reward[agent] = -1.0
            else:
                if self.mode == "human":
                    self.draw_trajectory(agent, action_dict[agent])
                self.all_agents_pos[agent] = np.array(new_positions[i])
                self.grid[new_positions[i][0], new_positions[i][1]] = 1
                done[agent] = False
                reward[agent] = 0.0
        # after all the position have been updated, we can retrieve agents
        # observations
        done_agents = 0
        for agent in done:
            if agent == "__all__":
                continue
            elif done[agent] is False:
                obs[agent] = self.get_agent_observation(agent)
                done_agents += 1
        if done_agents == 0:
            self.done["__all__"] = True

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

    def update_energy(self, agent, curr_act):
        """action_energy method.

        update the energy of a uav after taking an action, based on its current
        and previous action
        """
        if curr_act == 0:
            self.all_agents_charge[agent] -= self.energy_consumption["still"]
            return
        # if no prev action, the episode is has just begun
        if self.all_agents_prev_act[agent] is None:
            self.all_agents_charge[agent] -= \
                self.energy_consumption["different"]
        else:
            prev_act = self.all_agents_prev_act[agent]
            if prev_act == curr_act:
                self.all_agents_charge -= self.energy_consumption["same"]
            else:
                self.all_agents_charge -= self.energy_consumption["different"]

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
                return True

    def render(self, mode="human", screen_width=600, screen_height=400):
        """render method.

        Render the 2D scenario
        """
        self.mode = mode
        if mode == "human":
            # new elements are added in the bottom left corner. To center them,
            # translate by (screen_width/2, screen_height/2)
            self.screen_width = self.cell_size * (self.y+1)
            self.screen_height = self.cell_size * (self.x+1)
            # displacement represents the space outside the grid
            self.d_x = self.cell_size
            self.d_y = self.cell_size
            self.radius = self.cell_size/3
            background_width = self.screen_width
            background_height = self.screen_height
            # this is necessary otherwise it crates new
            if self.viewer is None:
                # if viewer is None, let's initialize the grid
                self.viewer = rendering.Viewer(
                    self.screen_width, self.screen_height)
                l, r, t, b = (
                    -background_width / 2,
                    background_width / 2,
                    background_height / 2,
                    -background_height / 2,
                )
                background = rendering.FilledPolygon(
                    [(l, b), (l, t), (r, t), (r, b)])
                backtrans = rendering.Transform(
                    translation=(self.screen_width / 2, self.screen_height / 2)
                )
                background.add_attr(backtrans)
                background.set_color(1.0, 1.0, 1.0)
                self.viewer.add_geom(background)
                # =============================================================
                # GRID
                # =============================================================
                for i in range(self.x+1):
                    row_line = rendering.Line(start=(self.cell_size,
                                                     i*self.cell_size,),
                                              end=(self.screen_width,
                                                   i*self.cell_size))
                    if i != self.x:
                        row_num = pyglet.text.Label(
                            str(i),
                            # font_name='Times New Roman',
                            font_size=14,
                            x=self.cell_size/2,
                            y=((self.x-1)-i)*self.cell_size +
                            self.cell_size/2,
                            anchor_x='center', anchor_y='center',
                            color=colors.BLACK_RGBA)
                        self.viewer.add_geom(DrawText(row_num))
                    self.viewer.add_geom(row_line)
                for j in range(self.y+1):
                    col_line = rendering.Line(start=(j*self.cell_size,
                                                     0),
                                              end=(j*self.cell_size,
                                                   self.screen_height -
                                                   self.cell_size))
                    if j != 0:
                        col_num = pyglet.text.Label(
                            str(j-1),
                            # font_name='Times New Roman',
                            font_size=14,
                            y=(self.x-1)*self.cell_size +
                            self.cell_size/2 + self.d_y,
                            x=j*self.cell_size + self.cell_size/2,
                            anchor_x='center', anchor_y='center',
                            color=colors.BLACK_RGBA)
                        self.viewer.add_geom(DrawText(col_num))
                    self.viewer.add_geom(col_line)
                # =============================================================
                # OBSTACLES
                # =============================================================
                for obstacle_pos in self.obstacles_pos:
                    obstacle_x, obstacle_y = obstacle_pos
                    obs_img = random.choice(paths.OBSTACLES)
                    obstacle = rendering.Image(obs_img, self.cell_size,
                                               self.cell_size)
                    obstacle_transf = rendering.Transform(
                        translation=(
                            obstacle_y*self.cell_size + self.cell_size/2 +
                            self.d_x, ((self.x-1)-obstacle_x)*self.cell_size +
                            self.cell_size/2
                        )
                    )
                    obstacle.add_attr(obstacle_transf)
                    self.viewer.add_geom(obstacle)
                # =============================================================
                # TARGETS
                # =============================================================
                for target_pos in self.targets_pos:
                    target_x, target_y = target_pos
                    # target = rendering.make_circle(radius)
                    target_img = paths.WIN_FLAG
                    target = rendering.Image(target_img, self.cell_size/2,
                                             self.cell_size/2)
                    target_transf = rendering.Transform(
                        translation=(
                            target_y*self.cell_size + self.cell_size/2 +
                            self.d_x, ((self.x-1)-target_x)*self.cell_size +
                            self.cell_size/2
                        )
                    )
                    target.add_attr(target_transf)
                    # target.set_color(*colors.TARGET_RGB)
                    self.viewer.add_geom(target)
            # =================================================================
            # UAVS AND UAVS_FOV
            # =================================================================
            for agent in self.all_agents_pos:
                agent_pos_x, agent_pos_y = self.all_agents_pos[agent]
                if [agent_pos_x, agent_pos_y] == [-1, -1]:
                    continue
                if self.max_charge - self.max_charge/3 \
                    < self.all_agents_charge[agent]\
                        <= self.max_charge:
                    uav_img = paths.UAV_HIGH_IMG
                elif self.max_charge - self.max_charge*2/3 \
                    < self.all_agents_charge[agent]\
                        <= self.max_charge - self.max_charge/3:
                    uav_img = paths.UAV_MID_IMG
                elif 0 <= self.all_agents_charge[agent] \
                        <= self.max_charge - self.max_charge*2/3:
                    uav_img = paths.UAV_LOW_IMG
                uav = rendering.Image(uav_img, self.cell_size, self.cell_size)
                l, r, t, b = (
                    -self.obs_y*self.cell_size - self.cell_size/2,
                    self.obs_y*self.cell_size + self.cell_size/2,
                    self.obs_x*self.cell_size + self.cell_size/2,
                    -self.obs_x*self.cell_size - self.cell_size/2,
                )
                uav_fov = rendering.FilledPolygon(
                    [(l, b), (l, t), (r, t), (r, b)])
                uav_fov_edges = rendering.PolyLine(
                    [(l, b), (l, t), (r, t), (r, b)], True)
                fov_trans = rendering.Transform(
                    translation=(
                        agent_pos_y*self.cell_size + self.cell_size/2 +
                        self.d_x, ((self.x-1)-agent_pos_x)*self.cell_size +
                        self.cell_size/2
                    )
                )
                uav_fov.add_attr(fov_trans)
                uav_fov_edges.add_attr(fov_trans)
                uav_fov._color.vec4 = colors.FOV
                uav_fov_edges.set_color(*colors.FOV_EDGES)
                uav_name = pyglet.text.Label(
                    agent[-1],
                    # font_name='Times New Roman',
                    font_size=14,
                    x=agent_pos_y*self.cell_size + self.cell_size/2 + self.d_x,
                    y=((self.x-1)-agent_pos_x)*self.cell_size +
                    self.cell_size/2,
                    anchor_x='center', anchor_y='center',
                    color=colors.WHITE_RGBA)
                uav_transf = rendering.Transform(
                    translation=(
                        agent_pos_y*self.cell_size + self.cell_size/2 +
                        self.d_x, ((self.x-1)-agent_pos_x)*self.cell_size +
                        self.cell_size/2
                    )
                )
                uav.add_attr(uav_transf)
                # one time since their position change at every timestep
                self.viewer.add_onetime(uav_fov_edges)
                self.viewer.add_onetime(uav_fov)
                self.viewer.add_onetime(uav)
                self.viewer.add_onetime(DrawText(uav_name))

            # =mode == "rgb_array")
            return self.viewer.render(return_rgb_array=True)

    def draw_trajectory(self, agent, action):
        """draw_trajectory method.

        It draws the trajectory of the uavs after they take an action inside
        the step function
        """
        agent_pos_x, agent_pos_y = self.all_agents_pos[agent]
        center_transf = rendering.Transform(
            translation=(
                agent_pos_y*self.cell_size + self.cell_size/2 + self.d_x,
                        ((self.x-1)-agent_pos_x)*self.cell_size +
                self.cell_size/2
            )
        )
        full_arrow = True
        traj_color = self.uavs_traj_color[agent]
        arrow_width = self.cell_size/10
        arrow_length = arrow_width*2
        if action == 0:
            print(f" agent {agent} didn't move")
        elif action == 1:
            top_arrow = rendering.PolyLine(
                [(-arrow_width, 0), (0, arrow_length), (arrow_width, 0)],
                full_arrow)
            top_arrow.add_attr(center_transf)
            top_arrow.set_color(*traj_color)
            # self.viewer.add_geom(top_arrow)
            self.viewer.add_onetime(top_arrow)
        elif action == 2:
            bot_arrow = rendering.PolyLine(
                [(-arrow_width, 0), (0, -arrow_length), (arrow_width, 0)],
                full_arrow)
            bot_arrow.add_attr(center_transf)
            bot_arrow.set_color(*traj_color)
            # self.viewer.add_geom(bot_arrow)
            self.viewer.add_onetime(bot_arrow)
        elif action == 3:
            left_arrow = rendering.PolyLine(
                [(0, arrow_width), (-arrow_length, 0), (0, -arrow_width)],
                full_arrow)
            left_arrow.add_attr(center_transf)
            left_arrow.set_color(*traj_color)
            # self.viewer.add_geom(left_arrow)
            self.viewer.add_onetime(left_arrow)
        elif action == 4:
            right_arrow = rendering.PolyLine(
                [(0, arrow_width), (arrow_length, 0), (0, -arrow_width)],
                full_arrow)
            right_arrow.add_attr(center_transf)
            right_arrow.set_color(*traj_color)
            # self.viewer.add_geom(right_arrow)
            self.viewer.add_onetime(right_arrow)
        else:
            print(f"invalid action: {action}")

    def close(self):
        """close method.

        Close the rendering view
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class DrawText:
    """DrawText class.

    class used to render text on pyglet window
    """

    def __init__(self, label: pyglet.text.Label):
        self.label = label

    def render(self):
        """render method.

        render the label
        """
        self.label.draw()


if __name__ == "__main__":
    print("hello")
