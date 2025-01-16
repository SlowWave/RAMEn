import os
import tomli
import shutil
import gymnasium as gym

from system_model import LorenzSystem

# get config data
with open(os.path.join(os.path.dirname(__file__),"config.toml"), "rb") as config_file:
    CFG = tomli.load(config_file)

class LorenzEnv(gym.Env):
    def __init__(self):
        pass
    
    def reset(self, *, seed = None, options = None):
        return super().reset(seed=seed, options=options)
    
    def step(self, action):
        return super().step(action)
    
    def render(self):
        return super().render()
    
    def close(self):
        return super().close()