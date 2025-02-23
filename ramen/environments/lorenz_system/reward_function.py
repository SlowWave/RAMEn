import os
import tomli
import numpy as np


# get config data
with open(os.path.join(os.path.dirname(__file__),"config.toml"), "rb") as config_file:
    CFG = tomli.load(config_file)


class RewardFunctionModel():
    def __init__(self):

        # define mapping dictionary
        self.reward_model_map = {
            "1": self._model_1,
            "2": self._model_2,
        }

        self.model_id = str(CFG["gymnasium"]["reward_function"]["model_id"])

    def get_reward(self, simulation_data):

        return self.reward_model_map[self.model_id](simulation_data)

    def _model_1(self, simulation_data):

        target_distance = 10
        slope = 5

        reward_1 = slope * np.pi ** 2 / 8 - slope * (np.atan(target_distance - simulation_data["observation"][:, -1][6])) ** 2

        is_last_reward = False
        rewards = [reward_1]

        return is_last_reward, rewards

    def _model_2(slef, simulation_data):

        r1 = 0
        r2 = 3
        r3 = -0.5

        is_last_reward = False

        rewards = [r1, r2, r3]

        return is_last_reward, rewards
