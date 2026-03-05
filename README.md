# Mass-Spring-Damper Simulation

This repository contains a simple modular Python simulation of a mass-spring-damper
system controlled via pole placement. The code is organized to isolate physics, simulation logic,
rendering, and control so that controllers (e.g. pole placement) can be implemented and adjusted without
major refactoring.

## Structure

- `physics.py` – Defines the plant dynamics.
- `simulation.py` – Time integration engine.
- `renderer.py` – `matplotlib` animation helper.
- `main.py` – Runs the animation

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
