"""
Double Pendulum Simulation
--------------------------
Day 1: Basic version created
Day 2: Added comments, docstring, and improved readability
Future: Explore Lorenz Attractor, Logistic Map, Duffing Oscillator
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -----------------------------
# Physical constants
# -----------------------------
g = 9.81     # Gravitational acceleration (m/s^2)
L1, L2 = 1, 1  # Lengths of pendulum arms (m)
m1, m2 = 1, 1  # Masses of pendulum bobs (kg)

# -----------------------------
# Initial conditions
# -----------------------------
theta1 = np.pi / 2   # Initial angle of pendulum 1 (radians)
theta2 = np.pi / 2   # Initial angle of pendulum 2 (radians)
omega1 = 0.0         # Initial angular velocity of pendulum 1
omega2 = 0.0         # Initial angular velocity of pendulum 2

# -----------------------------
# Simulation parameters
# -----------------------------
dt = 0.04      # Time step (s)
steps = 1500   # Number of simulation steps

# Arrays to store positions
x1_list, y1_list = [], []
x2_list, y2_list = [], []

# -----------------------------
# Equations of motion
# -----------------------------
def step(theta1, omega1, theta2, omega2, dt):
    """
    Compute one time step of the double pendulum system
    using the equations of motion.
    """
    delta = theta2 - theta1

    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
    den2 = (L2 / L1) * den1

    a1 = (m2 * L1 * omega1 ** 2 * np.sin(delta) * np.cos(delta)
          + m2 * g * np.sin(theta2) * np.cos(delta)
          + m2 * L2 * omega2 ** 2 * np.sin(delta)
          - (m1 + m2) * g * np.sin(theta1)) / den1

    a2 = (-m2 * L2 * omega2 ** 2 * np.sin(delta) * np.cos(delta)
          + (m1 + m2) * g * np.sin(theta1) * np.cos(delta)
          - (m1 + m2) * L1 * omega1 ** 2 * np.sin(delta)
          - (m1 + m2) * g * np.sin(theta2)) / den2

    omega1 += a1 * dt
    omega2 += a2 * dt
    theta1 += omega1 * dt
    theta2 += omega2 * dt

    return theta1, omega1, theta2, omega2

# -----------------------------
# Run the simulation
# -----------------------------
for _ in range(steps):
    theta1, omega1, theta2, omega2 = step(theta1, omega1, theta2, omega2, dt)

    # Convert polar coordinates → Cartesian for plotting
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    x1_list.append(x1)
    y1_list.append(y1)
    x2_list.append(x2)
    y2_list.append(y2)

# -----------------------------
# Visualization
# -----------------------------
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

line, = ax.plot([], [], 'o-', lw=2)

def animate(i):
    """
    Animation function: updates the pendulum position
    for frame i.
    """
    this_x = [0, x1_list[i], x2_list[i]]
    this_y = [0, y1_list[i], y2_list[i]]
    line.set_data(this_x, this_y)
    return line,

ani = animation.FuncAnimation(fig, animate, frames=steps, interval=40, blit=True)
plt.show()
