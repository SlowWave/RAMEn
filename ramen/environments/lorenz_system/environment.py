import os
import tomli
import gymnasium as gym
import numpy as np

from system_model import LorenzSystem
from observation_space import ObservationSpaceModel
from action_space import ActionSpaceModel
from reward_function import RewardFunctionModel

# get config data
with open(os.path.join(os.path.dirname(__file__),"config.toml"), "rb") as config_file:
    CFG = tomli.load(config_file)

class LorenzEnv(gym.Env):
    def __init__(self):

        self.epoch_time_horizon = CFG["gymnasium"]["epoch_time_horizon"]
        self.episode_time_step = CFG["dynamic_system"]["simulation"]["integration_step"]
        self.use_random_seed = CFG["gymnasium"]["use_random_seed"]
        self.random_seed = CFG["gymnasium"]["random_seed"]

        self.observation_space_model = ObservationSpaceModel()
        self.action_space_model = ActionSpaceModel()
        self.reward_function_model = RewardFunctionModel()        
        self.observation_space = self.observation_space_model.get_observation_space()
        self.action_space = self.action_space_model.get_action_space()

        self.system_model = LorenzSystem()
        self.simulation_data = None
        self.is_last_step = False

    def reset(self, *, seed = None, options = None):
        
        # set random seed if needed
        if self.use_random_seed:
            seed = self.random_seed
        
        super().reset(seed=seed, options=options)        
        
        # initialize Lorenz system
        state = self.system_model.set_initial_state()
        self._initialize_simulation_data(state)
    
        # get observation
        observation = self.observation_space_model.get_observation(self.simulation_data)

        info = {}
    
        return observation, info


    def step(self, action):
    
        # get elaborated action
        elaborated_action = self.action_space_model.get_elaborated_action(action, self.simulation_data)
    
        # update Lorenz system
        self._simulation_step(elaborated_action)
    
        # get observation
        observation = self.observation_space_model.get_observation(self.simulation_data)
    
        # get reward
        is_last_reward, rewards = self.reward_function_model.get_reward(self.simulation_data)
        reward = sum(rewards)
    
        # check termination condition
        if is_last_reward or self.is_last_step:
            terminated = True
        else:
            terminated = False

        # set truncated parameter to False (unused)
        truncated = False

        # store rewards values so that can be accessed by TensorboardCallback object
        info = {
            'rewards': rewards
        }
    
        return observation, reward, terminated, truncated, info
    
    def render(self):
        return
    
    def close(self):
        return
    
    def _simulation_step(self, input):
        
        # collect states
        state = [state[-1] for state in self.simulation_data["state"]]
        
        # get new state
        state = self.system_model.simulation_step(state, input)
        
        # upate simulation data
        self._update_simulation_data(state, input)
    
        # check termination condition
        if self.simulation_data["time"][-1] >= self.epoch_time_horizon:
            self.is_last_step = True
    

    def _initialize_simulation_data(self, state):
        
        self.simulation_data = {
            "state": state,
            "input": np.array([[0.0], [0.0], [0.0]]),
            "time": np.array([0.0]),
        }


    def _update_simulation_data(self, state, input, observation=None):
        
        self.simulation_data["state"] = np.concatenate(
            (self.simulation_data["state"], state),
            axis=1,
        )
        self.simulation_data["input"] = np.concatenate(
            (self.simulation_data["input"], input),
            axis=1,
        )
        self.simulation_data["time"] = np.append(
            self.simulation_data["time"],
            self.simulation_data["time"][-1] + self.episode_time_step,
        )