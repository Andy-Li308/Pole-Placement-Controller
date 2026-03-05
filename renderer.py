"""Rendering helpers for mass-spring-damper simulation.

The renderer uses ``matplotlib`` to animate the position of the mass over time,
display applied forces, pole locations, and displacement/force history.
The design keeps visualization separate from the physics so a controller can be
plugged in without touching plotting code.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


class MassSpringDamperRenderer:
    def __init__(self, t, states, forces, poles=None):
        """Prepare an animation given time, state, force, and pole histories.

        Parameters
        ----------
        t : ndarray
            Time vector.
        states : ndarray
            State history with shape ``(len(t), 2)``.
        forces : ndarray
            Force history with shape ``(len(t),)``.
        poles : ndarray, optional
            Closed-loop pole locations (complex array).
        """
        self.t = t
        self.states = states
        self.forces = forces
        self.poles = poles

        # Create figure with 4 subplots
        self.fig = plt.figure(figsize=(16, 5))
        
        # Left: main simulation (spring-mass-damper visual)
        self.ax_sim = self.fig.add_subplot(141)
        
        # Center-left: pole diagram
        self.ax_poles = self.fig.add_subplot(142)
        
        # Center-right: displacement vs time
        self.ax_disp = self.fig.add_subplot(143)
        
        # Right: force vs time
        self.ax_force = self.fig.add_subplot(144)
        
        # Setup main simulation axes
        x_vals = states[:, 0]
        span = np.ptp(x_vals) if x_vals.size > 0 else 0.0
        margin = 0.1 * (span if span > 0 else 1.0)
        self.x_min = -2.0
        self.x_max = 2.0
        
        self.ax_sim.set_xlim(self.x_min, self.x_max)
        self.ax_sim.set_ylim(-0.5, 0.5)
        self.ax_sim.set_xlabel("Position (m)")
        self.ax_sim.set_title("Mass-Spring-Damper System")
        self.ax_sim.grid(True, alpha=0.3)
        
        # Draw fixed wall on the left
        self.ax_sim.axvline(0, color='black', linewidth=3, label='Fixed Wall')
        
        # Spring line (will be updated)
        self.spring_line, = self.ax_sim.plot([], [], 'grey', linewidth=2, label='Spring')
        
        # Mass point
        self.mass_point, = self.ax_sim.plot([], [], 'bo', markersize=12, label='Mass')
        
        # Force arrow as a line with arrow marker
        self.force_arrow_line, = self.ax_sim.plot([], [], 'r-', linewidth=2.5, 
                                                   label='Force', marker='>', markersize=10)
        
        # Text info
        self.time_text = self.ax_sim.text(0.05, 0.95, "", transform=self.ax_sim.transAxes,
                                          verticalalignment='top', fontsize=10)
        self.force_text = self.ax_sim.text(0.05, 0.85, "", transform=self.ax_sim.transAxes,
                                           verticalalignment='top', fontsize=10)
        
        self.ax_sim.legend(loc='upper right', fontsize=8)
        
        # Setup pole diagram
        if poles is not None:
            self.ax_poles.scatter(poles.real, poles.imag, s=100, color='red', marker='x', linewidths=2)
            max_real = max(abs(poles.real.min()), abs(poles.real.max())) + 1
            max_imag = max(abs(poles.imag.min()), abs(poles.imag.max())) + 1 if np.any(poles.imag != 0) else 1
            self.ax_poles.set_xlim(-max_real, max_real)
            self.ax_poles.set_ylim(-max(max_imag, 0.5), max(max_imag, 0.5))
        else:
            self.ax_poles.set_xlim(-3, 1)
            self.ax_poles.set_ylim(-2, 2)
        
        # Draw real/imag axes for pole diagram
        self.ax_poles.axhline(0, color='k', linewidth=0.5)
        self.ax_poles.axvline(0, color='k', linewidth=0.5)
        self.ax_poles.set_xlabel("Real")
        self.ax_poles.set_ylabel("Imaginary")
        self.ax_poles.set_title("Closed-Loop Poles")
        self.ax_poles.grid(True, alpha=0.3)
        
        # Setup displacement vs time plot
        self.ax_disp.set_xlim(0, t[-1] if len(t) > 0 else 10)
        self.ax_disp.set_ylim(x_vals.min() - margin, x_vals.max() + margin)
        self.disp_line, = self.ax_disp.plot([], [], 'b-', linewidth=1.5, label='Displacement')
        self.ax_disp.set_xlabel("Time (s)")
        self.ax_disp.set_ylabel("Position (m)")
        self.ax_disp.set_title("Displacement vs Time")
        self.ax_disp.grid(True, alpha=0.3)
        self.ax_disp.legend(loc='upper right', fontsize=8)
        
        # Setup force vs time plot
        force_vals = forces if len(forces) > 0 else np.array([0.0])
        force_margin = 0.1 * (force_vals.max() - force_vals.min()) if np.ptp(force_vals) > 0 else 1.0
        self.ax_force.set_xlim(0, t[-1] if len(t) > 0 else 10)
        self.ax_force.set_ylim(force_vals.min() - force_margin, force_vals.max() + force_margin)
        self.force_line, = self.ax_force.plot([], [], 'r-', linewidth=1.5, label='Control Force')
        self.ax_force.set_xlabel("Time (s)")
        self.ax_force.set_ylabel("Force (N)")
        self.ax_force.set_title("Control Force vs Time")
        self.ax_force.grid(True, alpha=0.3)
        self.ax_force.legend(loc='upper right', fontsize=8)
        
        # For real-time synchronization
        self.wall_time_start = None
        self.sim_time_start = 0.0

    def _init(self):
        """Initialize animation artists."""
        self.spring_line.set_data([], [])
        self.mass_point.set_data([], [])
        self.force_arrow_line.set_data([], [])
        self.disp_line.set_data([], [])
        self.force_line.set_data([], [])
        self.time_text.set_text("")
        self.force_text.set_text("")
        return (self.spring_line, self.mass_point, self.force_arrow_line, 
                self.disp_line, self.force_line, self.time_text, self.force_text)

    def _update(self, frame):
        """Update animation for a given frame."""
        # Get current state
        x = self.states[frame, 0]
        current_time = self.t[frame]
        force = self.forces[frame] if frame < len(self.forces) else 0.0
        
        # Draw spring as a line from wall (x=0) to mass (x=x)
        self.spring_line.set_data([0, x], [0, 0])
        
        # Draw mass point
        self.mass_point.set_data([x], [0])
        
        # Draw force arrow as a line with marker
        # Only draw if force is significant
        if abs(force) > 0.01:
            force_scale = 0.15
            arrow_length = force * force_scale
            self.force_arrow_line.set_data([x, x + arrow_length], [0, 0])
        else:
            self.force_arrow_line.set_data([], [])
        
        # Update text
        self.time_text.set_text(f"t = {current_time:.2f} s")
        self.force_text.set_text(f"u = {force:.2f} N")
        
        # Update displacement vs time plot
        time_history = self.t[:frame + 1]
        disp_history = self.states[:frame + 1, 0]
        self.disp_line.set_data(time_history, disp_history)
        
        # Update force vs time plot
        force_history = self.forces[:frame + 1]
        self.force_line.set_data(time_history, force_history)
        
        return (self.spring_line, self.mass_point, self.force_arrow_line, 
                self.disp_line, self.force_line, self.time_text, self.force_text)

    def animate(self, interval=10, realtime=True):
        """Launch the animation.

        Parameters
        ----------
        interval : int
            Nominal delay between frames in milliseconds (if realtime=False).
        realtime : bool
            If True, synchronize animation to wall-clock time based on dt.
        """
        if realtime:
            # Create a simple frame counter that yields up to the max frame
            def frame_generator():
                wall_time_start = time.time()
                sim_time_start = self.t[0]
                frame = 0
                while frame < len(self.t) - 1:
                    elapsed_wall = time.time() - wall_time_start
                    target_sim_time = sim_time_start + elapsed_wall
                    
                    # Find frame index closest to target_sim_time
                    frame = min(np.searchsorted(self.t, target_sim_time, side='left'), 
                               len(self.t) - 1)
                    
                    yield frame
            
            ani = animation.FuncAnimation(
                self.fig,
                self._update,
                frames=frame_generator,
                init_func=self._init,
                interval=10,
                blit=True,
                repeat=False,
                cache_frame_data=False,
            )
        else:
            ani = animation.FuncAnimation(
                self.fig,
                self._update,
                frames=len(self.t),
                init_func=self._init,
                interval=interval,
                blit=True,
            )
        
        plt.tight_layout()
        plt.show()

