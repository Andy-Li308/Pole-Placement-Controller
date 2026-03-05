"""Entry point for running and visualizing a mass-spring-damper simulation.

This script sets up a simple plant with zero input and animates its free
response.  It is written to allow later injection of a controller function
that supplies ``u(t, x)``.
"""

import numpy as np
from physics import MassSpringDamper
from simulation import Simulation
from renderer import MassSpringDamperRenderer
from controller import Controller

def main():
    # plant parameters (SI units)
    m = 1.0      # kg
    k = 10.0      # N/m
    b = 1.0      # N*s/m

    plant = MassSpringDamper(m, k, b)

    #Pole placement controller 
    desired_poles = [0.1, 0.2]  # negative poles for stability
    controllerObject = Controller(m, k, b, desired_poles)
    controllerObject.place_poles()

#pass the control function to the simulation as the u function, which takes the current state and returns the control input
    u_func = controllerObject.control
    #u_func = lambda t, state: 0.0  # zero input for open-loop response; replace with controllerObject.control for closed-loop control

    #define the simulation params 
    sim = Simulation(plant, dt=0.005, u_func=u_func)

    t, x, u = sim.run(0.0, 30.0, x0=[1.0, 0.0]) # edit the initial condition and run time here as needed

    # Extract poles from controller for the pole diagram
    poles = controllerObject.K  # This will be updated to actual poles below
    try:
        poles = np.linalg.eigvals(plant.A - plant.B @ controllerObject.K)
    except:
        poles = controllerObject.desired_poles if hasattr(controllerObject, 'desired_poles') else None

    renderer = MassSpringDamperRenderer(t, x, u, poles=poles)
    # Real-time synchronization: animation runs at wall-clock pace
    renderer.animate(interval=10, realtime=True)


if __name__ == "__main__":
    main()
