# Goal: Add animation to visualize the system evolving over time
# Added: error plot, control signal plot, animated response

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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


# --- Pre-simulate full data ---
def run_simulation(kp, ki, kd, setpoint=1.0, dt=0.01, total_time=10.0):
    pid = PIDController(kp, ki, kd)
    t = np.arange(0, total_time, dt)
    y = 0.0
    outputs, errors, controls = [], [], []

    for _ in t:
        u = pid.compute(setpoint, y, dt)
        error = setpoint - y
        dydt = -y + u
        y += dydt * dt
        outputs.append(y)
        errors.append(error)
        controls.append(u)

    return t, outputs, errors, controls


t, outputs, errors, controls = run_simulation(kp=2.0, ki=0.5, kd=0.3)
setpoint = 1.0
STEP = 5  # animate every 5th frame for speed

# --- Figure layout ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
fig.suptitle('Day 3 — Animated PID Simulation', fontsize=13, fontweight='bold')

line_out,  = ax1.plot([], [], color='royalblue', linewidth=2, label='Output')
ax1.axhline(setpoint, color='gray', linestyle='--', linewidth=1.2, label='Setpoint')
ax1.set_ylabel('System Output')
ax1.set_ylim(-0.2, 1.6)
ax1.legend(loc='lower right', fontsize=8)
ax1.grid(True, alpha=0.4)

line_err,  = ax2.plot([], [], color='tomato', linewidth=1.8, label='Error')
ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
ax2.set_ylabel('Error')
ax2.set_ylim(-0.5, 1.2)
ax2.legend(loc='upper right', fontsize=8)
ax2.grid(True, alpha=0.4)

line_ctrl, = ax3.plot([], [], color='seagreen', linewidth=1.8, label='Control Signal (u)')
ax3.axhline(0, color='gray', linestyle='--', linewidth=1)
ax3.set_ylabel('Control Signal')
ax3.set_xlabel('Time (s)')
ax3.set_ylim(-1, 5)
ax3.legend(loc='upper right', fontsize=8)
ax3.grid(True, alpha=0.4)

for ax in (ax1, ax2, ax3):
    ax.set_xlim(0, t[-1])

frames = range(0, len(t), STEP)


def animate(i):
    line_out.set_data(t[:i],  outputs[:i])
    line_err.set_data(t[:i],  errors[:i])
    line_ctrl.set_data(t[:i], controls[:i])
    return line_out, line_err, line_ctrl


ani = animation.FuncAnimation(
    fig, animate, frames=frames,
    interval=20, blit=True, repeat=False
)

plt.tight_layout()
plt.show()
