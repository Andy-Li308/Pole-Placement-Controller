"""Simulation engine for running the mass-spring-damper plant.

This module wraps a physics object and integrates it forward in time using a
simple explicit integrator. The structure is kept minimal so that control
logic can be injected easily later.
"""

import numpy as np
from physics import MassSpringDamper, default_input


class Simulation:
    """Run a time-domain simulation of the plant.

    Parameters
    ----------
    plant : MassSpringDamper
        The physical system to simulate.
    dt : float
        Integration time step (s).
    u_func : callable
        Function ``u(t, state)`` returning the scalar force input at time
        ``t`` and state ``state``. Defaults to zero.
    """

    def __init__(self, plant: MassSpringDamper, dt: float, u_func=None):
        self.plant = plant
        self.dt = dt
        self.u_func = u_func if u_func is not None else default_input

    def run(self, t0: float, tf: float, x0):
        """Simulate from ``t0`` to ``tf`` starting with initial state ``x0``.

        Returns
        -------
        t : ndarray
            Time vector.
        x : ndarray
            State history with shape ``(len(t), 2)``.
        u : ndarray
            Input force history with shape ``(len(t),)``.
        """
        t = np.arange(t0, tf + self.dt, self.dt)
        x = np.zeros((len(t), 2))
        u = np.zeros(len(t))
        x[0] = x0
        for idx in range(1, len(t)):
            ti = t[idx - 1]
            xi = x[idx - 1]
            u_val = self.u_func(ti, xi)
            u[idx - 1] = u_val
            dx = self.plant.derivative(ti, xi, u_val)
            x[idx] = xi + self.dt * np.asarray(dx)
        u[-1] = self.u_func(t[-1], x[-1])  # capture final force
        return t, x, u
