import os
import tomli
import numpy as np
from gymnasium import spaces

# get config data
with open(os.path.join(os.path.dirname(__file__),"config.toml"), "rb") as config_file:
    CFG = tomli.load(config_file)


class ActionSpaceModel():
    def __init__(self):

        # define mapping dictionaries
        self.action_model_map = {
            "1": self._model_1,
            "2": self._model_2,
        }

        self.action_elaboration_map = {
            "1": self._elaborate_action_1,
            "2": self._elaborate_action_2,
        }

        self.model_id = str(CFG["gymnasium"]["action_space"]["model_id"])

    def get_action_space(self):

        if self.model_id not in self.action_model_map:
            raise ValueError(f"Unsupported model_id: {self.model_id} for action space")
        
        return self.action_model_map[self.model_id]()
    
    def get_elaborated_action(self, action, storage):

        if self.model_id not in self.action_elaboration_map:
            raise ValueError(f"Unsupported model_id: {self.model_id} for action space")

        return self.action_elaboration_map[self.model_id](action, storage)

    def _model_1(self):

        # define action space limits
        action_limit = np.array(
            [0.5, 0.5, 0.5],
            dtype=np.float32,
        )

        # define action space
        action_space = spaces.Box(
            -action_limit,
            action_limit,
            dtype=np.float32
        )

        return action_space

    def _elaborate_action_1(self, action, simulation_data):

        return action

    def _model_2(self):

        # define action space
        action_space = spaces.MultiDiscrete(
            nvec=np.array([21, 21, 21])
        )

        return action_space

    def _elaborate_action_2(self, action, simulation_data):
        
        # map raw action in the interval [-0.5, 0.5]
        slope = 0.05
        y_intercept = - 0.5

        return action * slope + y_intercept
