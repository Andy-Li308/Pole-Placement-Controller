"""Physics definitions for the mass-spring-damper plant.

The system is defined by the second-order differential equation::

    m x'' + b x' + k x = u

which can be written in state-space form with state [x, x'].  

This module exposes a lightweight class that computes the state derivative
and can optionally be extended with control inputs.
"""

import numpy as np

default_input = lambda t, state: 0.0


class MassSpringDamper:
    """Simple mass-spring-damper plant.

    Attributes
    ----------
    m : float
        Mass (kg).
    k : float
        Spring constant (N/m).
    b : float
        Damping coefficient (N·s/m).
    """

    def __init__(self, m: float, k: float, b: float):
        self.m = m
        self.k = k
        self.b = b
        
        # State-space matrices for x' = A*x + B*u
        # state = [position, velocity]
        self.A = np.array([[0.0, 1.0],
                           [-k/m, -b/m]])
        self.B = np.array([[0.0],
                           [1.0/m]])

    def derivative(self, t: float, state, u: float = None):
        """Return state derivative given time, state, and input force.

        Parameters
        ----------
        t : float
            Current time (s).
        state : array_like
            Current state vector ``[x, x_dot]``.
        u : float, optional
            Input force (N). If omitted, zero force is assumed.

        Returns
        -------
        ndarray
            Time derivative of the state ``[x_dot, x_ddot]``.
        """
        x, x_dot = state
        if u is None:
            u = 0.0
        x_ddot = (u - self.b * x_dot - self.k * x) / self.m
        return [x_dot, x_ddot]
