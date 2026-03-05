# Mass-Spring-Damper Simulation

This repository contains a simple modular Python simulation of a mass-spring-damper
system. The code is organized to isolate physics, simulation logic, and
rendering so that controllers (e.g. pole placement) can be added later without
major refactoring.

## Structure

- `physics.py` – Defines the plant dynamics.
- `simulation.py` – Time integration engine.
- `renderer.py` – `matplotlib` animation helper.
- `main.py` – Example usage, runs a free response and shows animation.

## Usage

1. Create a Python virtual environment and install dependencies:

   ```sh
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Run the simulation:

   ```sh
   python main.py
   ```

The architecture makes it simple to replace `u_func` in `main.py` with a
controller that computes forces based on the state.
