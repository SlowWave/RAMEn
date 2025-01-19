import os
import tomli
import numpy as np
from gymnasium import spaces

# get config data
with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),"config.toml"), "rb") as config_file:
    CFG = tomli.load(config_file)



class ObservationSpaceModel():
    def __init__(self):

        # define mapping dictionary
        self.observation_model_map = {
            "1": self._observation_model_1,
            "2": self._observation_model_2,
        }
        
        self.observation_map = {
            "1": self._observation_1,
            "2": self._observation_2,
        }

        self.model_id = str(CFG["gymnasium"]["observation_space"]["model_id"])

    def get_observation_space(self):

        if self.model_id not in self.observation_model_map:
            raise ValueError(f"Unsupported model_id: {self.model_id} for obsevation space")
        
        return self.observation_model_map[self.model_id]()

    def get_observation(self):

        if self.model_id not in self.observation_model_map:
            raise ValueError(f"Unsupported model_id: {self.model_id} for obsevation space")
        
        return self.observation_model_map[self.model_id]()

    def _observation_model_1(self):

        # define observation space limits
        observation_limit = np.array(
            [
                np.finfo(np.float32).max,   # state [0]
                np.finfo(np.float32).max,   # state [1]
                np.finfo(np.float32).max,   # state [2]
                np.finfo(np.float32).max,   # state difference [0]
                np.finfo(np.float32).max,   # state difference [1]
                np.finfo(np.float32).max,   # state difference [2]
                np.finfo(np.float32).max,   # distance from eq point 1 [0]
                np.finfo(np.float32).max,   # distance from eq point 1 [1]
                np.finfo(np.float32).max,   # distance from eq point 1 [2]
                np.finfo(np.float32).max,   # distance from eq point 2 [0]
                np.finfo(np.float32).max,   # distance from eq point 2 [1]
                np.finfo(np.float32).max,   # distance from eq point 2 [2]
            ],
            dtype=np.float32,
        )

        # define observation space
        observation_space = spaces.Box(
            -observation_limit,
            observation_limit,
            dtype=np.float32
        )

        return observation_space

    def _observation_1(self, state):

        obs = state
        return obs

    def _observation_model_2(self):

        # define observation space limits
        observation_limit = np.array(
            [
                1,                          # mrp tracking error [0]
                1,                          # mrp tracking error [1]
                1,                          # mrp tracking error [2]
                np.finfo(np.float32).max,   # omega tracking error [0]
                np.finfo(np.float32).max,   # omega tracking error [1]
                np.finfo(np.float32).max,   # omega tracking error [2]
                np.finfo(np.float32).max,   # feedback control signal [0]
                np.finfo(np.float32).max,   # feedback control signal [1]
                np.finfo(np.float32).max,   # feedback control signal [2]
                0.5,                        # last rl agent action [0]
                0.5,                        # last rl agent action [1]
                0.5,                        # last rl agent action [2]
            ],
            dtype=np.float32,
        )

        # define observation space
        observation_space = spaces.Box(
            -observation_limit,
            observation_limit,
            dtype=np.float32
        )

        return observation_space
    
    
    def _observation_2(self, state):
        
        obs = state
        return obs