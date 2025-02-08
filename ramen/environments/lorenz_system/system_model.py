import os
import sys
import tomli
import math

# add parent directory to "sys.path" to import modules from that path
sys.path.append(os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir))

from modules.dynamic_system import DynamicSystem

# get config data
with open(os.path.join(os.path.dirname(__file__),"config.toml"), "rb") as config_file:
    CFG = tomli.load(config_file)


class LorenzSystem(DynamicSystem):
    def __init__(self):
        
        super(LorenzSystem, self).__init__()
        
        # general attributes
        self.sigma = CFG["dynamic_system"]["sigma"]
        self.rho = CFG["dynamic_system"]["rho"]
        self.beta = CFG["dynamic_system"]["beta"]
        self.state_dim = CFG["dynamic_system"]["state_dim"]
        self.input_dim = CFG["dynamic_system"]["input_dim"]
        self.state_boundaries = CFG["dynamic_system"]["state_boundaries"]
        self.input_boundaries = CFG["dynamic_system"]["input_boundaries"]
        self.tag = "LorenzSystem"
        
        # simulation attributes
        self.integration_step = CFG["dynamic_system"]["simulation"]["integration_step"]

        # set equilibrium points
        self.equilibrium_points = self._get_equlibrium_points()

    def ode(self, t, x, u):
        """
        Computes the derivative of the state variables of the Lorenz system.

        Args:
            t (float): The current time.
            x (list): The current state variables of the system.
            u (list): The current control input.

        Returns:
            list: The derivative of the state variables.
        """

        x_dot_1 = self.sigma * (x[1] - x[0]) + u[0]
        x_dot_2 = x[0] * (self.rho - x[2]) - x[1] + u[1]
        x_dot_3 = x[0] * x[1] - self.beta * x[2] + u[2]

        x_dot = [x_dot_1, x_dot_2, x_dot_3]

        return x_dot
    
    def _get_equlibrium_points(self):
        """
        Returns the equilibrium points of the Lorenz system

        Returns:
            list: List of equilibrium points
        """
        
        eq_1 = [0.0, 0.0, 0.0]
        eq_2 = [
            math.sqrt(self.beta * (self.rho - 1.0)),
            math.sqrt(self.beta * (self.rho - 1.0)),
            self.rho - 1.0
        ]
        eq_3 = [
            -math.sqrt(self.beta * (self.rho - 1.0)),
            -math.sqrt(self.beta * (self.rho - 1.0)),
            self.rho - 1.0
        ]
        
        return [eq_1, eq_2, eq_3]
    
if __name__ == "__main__":
    lorenz_system = LorenzSystem()
    lorenz_system.simulate_system(100)