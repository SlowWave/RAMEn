import os
import sys

# add parent directory to "sys.path" to import modules from that path
sys.path.append(os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir))

from modules.dynamic_system import DynamicSystem


class LorenzSystem(DynamicSystem):
    def __init__(self, sigma=10, rho=28, beta=8/3):
        
        super(LorenzSystem, self).__init__()
        
        # initialize attributes
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

        self.tag = "LorenzSystem"
        self.state_boundaries = [
            [-20, -20, 0],
            [20, 20, 40],
        ]

        self.input_boundaries = [
            [-20, -20, 0],
            [20, 20, 40],
        ]
        
        self.state_dim = 3
        self.input_dim = 3
        self.integration_step = 0.01

    def ode(self, t, x, u):
        """
        Calculates the derivative of the state variables of the dynamic system.

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
    
    
if __name__ == "__main__":
    lorenz_system = LorenzSystem()
    lorenz_system.simulate_system(100, use_inputs=False)