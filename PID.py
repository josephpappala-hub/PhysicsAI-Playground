import numpy as np
import matplotlib.pyplot as plt

# --- PID Controller Class ---
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain

        self.prev_error = 0
        self.integral = 0

    def compute(self, setpoint, measured, dt):
        error = setpoint - measured

        # Proportional term
        P = self.kp * error

        # Integral term
        self.integral += error * dt
        I = self.ki * self.integral

        # Derivative term
        derivative = (error - self.prev_error) / dt
        D = self.kd * derivative

        self.prev_error = error

        return P + I + D


# Simple System
# system response: dy/dt = -y + u (where u is control input)

def simulate(kp, ki, kd, setpoint=1.0, dt=0.01, total_time=10.0):
    pid = PIDController(kp, ki, kd)

    t = np.arange(0, total_time, dt)
    y = 0.0  # initial output
    output_history = []
    setpoint_history = []

    for _ in t:
        u = pid.compute(setpoint, y, dt)

        # First-order system dynamics
        dydt = -y + u
        y += dydt * dt

        output_history.append(y)
        setpoint_history.append(setpoint)

    return t, output_history, setpoint_history


# Plotting
kp, ki, kd = 2.0, 0.5, 0.1
t, output, setpoint = simulate(kp, ki, kd)

plt.figure(figsize=(10, 5))
plt.plot(t, output, label='System Output', color='royalblue')
plt.plot(t, setpoint, 'r--', label='Setpoint', linewidth=1.5)
plt.title('Day 1 — Basic PID Response')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
