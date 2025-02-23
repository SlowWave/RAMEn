import os
import tomli
import gymnasium as gym
import numpy as np

from system_model import LorenzSystem
from observation_space import ObservationSpaceModel
from action_space import ActionSpaceModel
from reward_function import RewardFunctionModel

# get config data
with open(os.path.join(os.path.dirname(__file__), "config.toml"), "rb") as config_file:
    CFG = tomli.load(config_file)


class LorenzEnv(gym.Env):
    def __init__(self):

        # general attributes
        self.epoch_time_horizon = CFG["gymnasium"]["epoch_time_horizon"]
        self.episode_time_step = CFG["dynamic_system"]["simulation"]["integration_step"]
        self.use_random_seed = CFG["gymnasium"]["use_random_seed"]
        self.random_seed = CFG["gymnasium"]["random_seed"]

        # gymnasium attributes
        self.observation_space_model = ObservationSpaceModel()
        self.action_space_model = ActionSpaceModel()
        self.reward_function_model = RewardFunctionModel()
        self.observation_space = self.observation_space_model.get_observation_space()
        self.action_space = self.action_space_model.get_action_space()

        # system attributes
        self.system_model = LorenzSystem()
        self.simulation_data = None
        self.is_last_step = False

    def reset(self, *, seed=None, options=None):

        # set random seed if needed
        if self.use_random_seed:
            seed = self.random_seed

        super().reset(seed=seed, options=options)

        # initialize Lorenz system
        state = self.system_model.set_initial_state()
        self.simulation_data = dict()
        self._initialize_simulation_data(state=state)

        # get observation
        observation = self.observation_space_model.get_observation(self.simulation_data)
        self._initialize_simulation_data(observation=observation)

        info = {}

        return observation, info

    def step(self, action):

        # get elaborated action
        elaborated_action = self.action_space_model.get_elaborated_action(
            action, self.simulation_data
        )

        # update Lorenz system
        self._simulation_step(elaborated_action)

        # get observation
        observation = self.observation_space_model.get_observation(self.simulation_data)

        # get reward
        is_last_reward, rewards = self.reward_function_model.get_reward(
            self.simulation_data
        )
        reward = sum(rewards)

        # update simulation data
        self._update_simulation_data(reward=reward, observation=observation)

        # check termination condition
        if is_last_reward or self.is_last_step:
            terminated = True
        else:
            terminated = False

        # set truncated parameter to False (unused)
        truncated = False

        # store rewards values so that can be accessed by TensorboardCallback object
        info = {"rewards": rewards}

        return observation, reward, terminated, truncated, info

    def render(self):
        return

    def close(self):
        return

    def _simulation_step(self, input):

        # collect last system state
        state = [state[-1] for state in self.simulation_data["state"]]

        # perform simulation step
        state = self.system_model.simulation_step(state, input)

        # upate simulation data state, input and time
        self._update_simulation_data(time=True, state=state, input=input)

        # check termination condition
        if self.simulation_data["time"][-1] >= self.epoch_time_horizon:
            self.is_last_step = True

    def _initialize_simulation_data(self, state=None, observation=None):

        # set time, input, reward and equilibrium points data
        self.simulation_data["time"] = np.array([0.0])
        self.simulation_data["input"] = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)
        self.simulation_data["reward"] = np.array([0.0])
        self.simulation_data["equilibrium_points"] = (
            self.system_model.equilibrium_points
        )

        if state is not None:
            self.simulation_data["state"] = np.array(state).reshape(-1, 1)

        if observation is not None:
            self.simulation_data["observation"] = np.array(observation).reshape(-1, 1)

    def _update_simulation_data(
        self, time=False, state=None, input=None, observation=None, reward=None
    ):

        if time:
            self.simulation_data["time"] = np.append(
                self.simulation_data["time"],
                self.simulation_data["time"][-1] + self.episode_time_step,
            )

        if state is not None:
            self.simulation_data["state"] = np.concatenate(
                (self.simulation_data["state"], np.array(state).reshape(-1, 1)),
                axis=1,
            )

        if input is not None:
            self.simulation_data["input"] = np.concatenate(
                (self.simulation_data["input"], np.array(input).reshape(-1, 1)),
                axis=1,
            )

        if observation is not None:
            self.simulation_data["observation"] = np.concatenate(
                (
                    self.simulation_data["observation"],
                    np.array(observation).reshape(-1, 1),
                ),
                axis=1,
            )

        if reward is not None:
            self.simulation_data["reward"] = np.append(
                self.simulation_data["reward"],
                reward,
            )


if __name__ == "__main__":

    # create environment
    lorenz_env = LorenzEnv()

    # reset environment
    lorenz_env.reset()

    # for loop
    for i in range(10):

        # step environment
        lorenz_env.step(np.array([0.0, 0.0, 0.0]))
