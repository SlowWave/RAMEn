import math
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

class DynamicSystem:
    def __init__(self):
        """
        Base class for all other dynamic systems classes.

        Args:
            None
        """

        # initialize attributes
        self.state_boundaries = None
        self.input_boundaries = None
        self.state_dim = None
        self.input_dim = None
        self.integration_step = None

    def set_initial_state(self, state=None):
        """
        Sets the initial state of the dynamic system.

        Args:
            state (list or None): The initial state of the system. If None, a random initial state is generated.

        Returns:
            list: The initial state of the system.
        """

        if not state:
            initial_state = list()
            for idx in range(self.state_dim):
                initial_state.append(
                    random.uniform(
                        self.state_boundaries[0][idx],
                        self.state_boundaries[1][idx],
                    )
                )

        else:
            initial_state = state

        return initial_state

    def generate_input_signals(self, use_inputs, inputs_shape, inputs_num):
        """
        Generates input signals for the dynamic system.

        Args:
            use_inputs (bool): Flag indicating whether to use inputs or not.
            inputs_shape (str): Shape of the inputs. If 'random', generates random inputs.
            inputs_num (int): Number of inputs to generate.

        Returns:
            list: List of input signals. Each input signal is a numpy array of shape (inputs_num,).
        """

        input_signals = list()

        if not use_inputs:
            for _ in range(self.input_dim):
                input_signals.append(np.zeros(inputs_num))

        elif inputs_shape == "random":
            for idx in range(self.input_dim):
                input_signals.append(
                    np.random.uniform(
                        self.input_boundaries[0][idx],
                        self.input_boundaries[1][idx],
                        inputs_num,
                    )
                )

        return input_signals

    def ode(self):
        pass

    def simulate_system(
        self,
        time_horizon,
        initial_state=None,
        use_inputs=False,
        inputs_shape="random",
    ):
        """
        Simulates the dynamics of the dynamic system over a given time horizon and plots the results.

        Args:
            time_horizon (float): The duration of the simulation in seconds.
            initial_state (list, optional): The initial state of the system. If None, a random initial state is generated. Defaults to None.
            use_inputs (bool, optional): Flag indicating whether to use inputs or not. Defaults to False.
            inputs_shape (str, optional): The shape of the inputs. If 'random', generates random inputs. Defaults to "random".
        """

        # propagate system dynamics
        data_dict = self.propagate_states(
            time_horizon,
            initial_state,
            use_inputs,
            inputs_shape,
        )

        fig, axes = plt.subplots(len(data_dict["states"]))

        for i, state in enumerate(data_dict["states"]):
            axes[i].plot(
                data_dict["time_steps"],
                state,
                label=f"$x_{i}$",
                color=f"C{i}",
            )
            axes[i].grid()
            axes[i].set_xlabel("Time [s]")
            axes[i].set_ylabel(f"$x_{i}$")
        
        fig.tight_layout()
        fig.legend()
        # plt.show()
        
        if use_inputs:

            fig, axes = plt.subplots(len(data_dict["inputs"]))

            for i, input_signal in enumerate(data_dict["inputs"]):
                axes[i].plot(
                    data_dict["time_steps"],
                    input_signal,
                    label=f"$u_{i}$",
                    color=f"C{i}",
                )
                axes[i].grid()
                axes[i].set_xlabel("Time [s]")
                axes[i].set_ylabel(f"$u_{i}$")
        fig.tight_layout()
        fig.legend()
        plt.show()
        
        self.display_animation(data_dict)

    def propagate_states(
        self,
        time_horizon,
        initial_state,
        use_inputs,
        inputs_shape,
    ):
        """
        Propagates the states of the dynamic system over a given time horizon.

        Args:
            time_horizon (float): The duration of the simulation in seconds.
            initial_state (list): The initial state of the system.
            use_inputs (bool): Flag indicating whether to use inputs or not.
            inputs_shape (str): The shape of the inputs.

        Returns:
            dict: A dictionary containing the time steps, states, and inputs.
                - time_steps (ndarray): An array of time steps.
                - states (list): A list of state arrays.
                - inputs (list): A list of input arrays.
        """

        # define timing parameters
        num_steps = math.ceil(time_horizon / self.integration_step)
        time_horizon = num_steps * self.integration_step
        time_steps = np.linspace(0.0, time_horizon, num_steps + 1)

        # set dynamic system states and inputs
        current_state = self.set_initial_state(initial_state)
        states = [[state] for state in current_state]
        inputs = self.generate_input_signals(use_inputs, inputs_shape, num_steps + 1)

        # propagate system states
        for step in range(len(time_steps) - 1):
            
            # update system state
            current_state = self.simulation_step(
                state=current_state,
                input=[u[step] for u in inputs],
                time=time_steps[step],
            )

            # update states list
            for idx, state in enumerate(states):
                state.append(current_state[idx])

        data_dict = dict(time_steps=time_steps, states=states, inputs=inputs)

        return data_dict
    
    def simulation_step(
        self,
        state,
        input,
        time=0.0,
        method="RK45",
        dense_output=False,
    ):
        """_summary_

        Args:
            state (_type_): _description_
            input (_type_): _description_
            time (float, optional): _description_. Defaults to 0.0.
            method (str, optional): _description_. Defaults to "RK45".
            dense_output (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        # integration step
        ode_solution = solve_ivp(
            fun=self.ode,
            t_span=(time, time + self.integration_step),
            y0=state,
            method=method,
            dense_output=dense_output,
            args=(input,),
        )

        # get updated state
        updated_state = [state[-1] for state in ode_solution.y]

        return updated_state
    
    def display_animation(self, data_dict):
        
        # Extract state data
        x_data, y_data, z_data = data_dict["states"]
        time_steps = data_dict["time_steps"]

        # Set up the figure and 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(min(x_data), max(x_data))
        ax.set_ylim(min(y_data), max(y_data))
        ax.set_zlim(min(z_data), max(z_data))
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$z$")
        ax.set_title("3D Animation of State Variables")

        # Initialize line and point
        line, = ax.plot([], [], [], lw=2, label="Trajectory")
        point, = ax.plot([], [], [], 'ro', label="Current Position")

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            return line, point

        def update(frame):
            # Update the trajectory and current position
            line.set_data(x_data[:frame], y_data[:frame])
            line.set_3d_properties(z_data[:frame])
            point.set_data([x_data[frame]], [y_data[frame]])  # Wrap in a list
            point.set_3d_properties([z_data[frame]])         # Wrap in a list
            return line, point

        anim = animation.FuncAnimation(
            fig, update, frames=len(time_steps), init_func=init, blit=False, interval=0.001
        )

        plt.legend()
        plt.show()