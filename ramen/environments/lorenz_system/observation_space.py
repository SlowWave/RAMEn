import os
import tomli
import numpy as np
from gymnasium import spaces

# get config data
with open(os.path.join(os.path.dirname(__file__),"config.toml"), "rb") as config_file:
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

    def get_observation(self, simulation_data):

        if self.model_id not in self.observation_model_map:
            raise ValueError(f"Unsupported model_id: {self.model_id} for obsevation space")
        
        return self.observation_map[self.model_id](simulation_data)

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
                np.finfo(np.float32).max,   # distance from eq point
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

    def _observation_1(self, simulation_data):

        # get states
        state_1 = simulation_data["state"][:, -1][0]
        state_2 = simulation_data["state"][:, -1][1]
        state_3 = simulation_data["state"][:, -1][2]

        # get state differences
        if simulation_data["state"].shape[1] > 1:
            state_diff_1 = state_1 - simulation_data["state"][:, -2][0]
            state_diff_2 = state_2 - simulation_data["state"][:, -2][1]
            state_diff_3 = state_3 - simulation_data["state"][:, -2][2]

        else:
            state_diff_1 = 0.0
            state_diff_2 = 0.0
            state_diff_3 = 0.0

        # get distance from equilibrium point
        state = np.array([state_1, state_2, state_3])
        eq_point = np.array(simulation_data["equilibrium_points"][1])
        d_eq = np.linalg.norm(state - eq_point)

        obseration = np.array(
            [
                state_1,
                state_2,
                state_3,
                state_diff_1,
                state_diff_2,
                state_diff_3,
                d_eq,
            ],
            dtype=np.float32,
        )

        return obseration

    # * Note: the following method is a copy of _observation_model_1, please modify if needed
    def _observation_model_2(self):

        # define observation space limits
        observation_limit = np.array(
            [
                np.finfo(np.float32).max,   # state [0]
                np.finfo(np.float32).max,   # state [1]
                np.finfo(np.float32).max,   # state [2]
                np.finfo(np.float32).max,   # state difference [0]
                np.finfo(np.float32).max,   # state difference [1]
                np.finfo(np.float32).max,   # state difference [2]
                np.finfo(np.float32).max,   # distance from eq point
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

    # * Note: the following method is a copy of _observation_1, please modify if needed
    def _observation_2(self, simulation_data):

        # get states
        state_1 = simulation_data["state"][:, -1][0]
        state_2 = simulation_data["state"][:, -1][1]
        state_3 = simulation_data["state"][:, -1][2]

        # get state differences
        if simulation_data["state"].shape[1] > 1:
            state_diff_1 = state_1 - simulation_data["state"][:, -2][0]
            state_diff_2 = state_2 - simulation_data["state"][:, -2][1]
            state_diff_3 = state_3 - simulation_data["state"][:, -2][2]

        else:
            state_diff_1 = 0.0
            state_diff_2 = 0.0
            state_diff_3 = 0.0

        # get distance from equilibrium point
        state = np.array([state_1, state_2, state_3])
        eq_point = np.array(simulation_data["equilibrium_points"][1])
        d_eq = np.linalg.norm(state - eq_point)

        obseration = np.array(
            [
                state_1,
                state_2,
                state_3,
                state_diff_1,
                state_diff_2,
                state_diff_3,
                d_eq,
            ],
            dtype=np.float32,
        )

        return obseration
