# Goal: Visualize how changing Kp, Ki, Kd affects the system response
# Added: multiple subplots comparing underdamped, overdamped, and tuned responses

import numpy as np
import matplotlib.pyplot as plt


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, setpoint, measured, dt):
        error = setpoint - measured
        P = self.kp * error
        self.integral += error * dt
        I = self.ki * self.integral
        derivative = (error - self.prev_error) / dt
        D = self.kd * derivative
        self.prev_error = error
        return P + I + D

    def reset(self):
        self.prev_error = 0
        self.integral = 0


def simulate(kp, ki, kd, setpoint=1.0, dt=0.01, total_time=10.0):
    pid = PIDController(kp, ki, kd)
    t = np.arange(0, total_time, dt)
    y = 0.0
    output_history = []

    for _ in t:
        u = pid.compute(setpoint, y, dt)
        dydt = -y + u
        y += dydt * dt
        output_history.append(y)

    return t, output_history


# --- Different tuning configurations ---
configs = [
    {"label": "High Kp (Aggressive)",   "kp": 8.0,  "ki": 0.1, "kd": 0.0, "color": "tomato"},
    {"label": "Low Kp (Sluggish)",       "kp": 0.5,  "ki": 0.1, "kd": 0.0, "color": "orange"},
    {"label": "No Derivative (Overshoot)","kp": 2.0, "ki": 1.0, "kd": 0.0, "color": "orchid"},
    {"label": "Well Tuned",              "kp": 2.0,  "ki": 0.5, "kd": 0.3, "color": "seagreen"},
]

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
axes = axes.flatten()
setpoint = 1.0

for i, cfg in enumerate(configs):
    t, output = simulate(cfg["kp"], cfg["ki"], cfg["kd"])
    axes[i].plot(t, output, color=cfg["color"], linewidth=2, label='Output')
    axes[i].axhline(setpoint, color='gray', linestyle='--', linewidth=1.2, label='Setpoint')
    axes[i].set_title(f'{cfg["label"]}\nKp={cfg["kp"]}  Ki={cfg["ki"]}  Kd={cfg["kd"]}', fontsize=10)
    axes[i].set_xlabel('Time (s)')
    axes[i].set_ylabel('Output')
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.4)
    axes[i].set_ylim(-0.2, 1.8)

fig.suptitle('Day 2 — PID Gain Tuning Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
