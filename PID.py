# Goal: Replace abstract first-order system with a real physical model
# Added: mass-spring-damper dynamics, matplotlib slider widgets for live tuning

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec


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
        D = self.kd * (error - self.prev_error) / dt
        self.prev_error = error
        return P + I + D

    def reset(self):
        self.prev_error = 0
        self.integral = 0


# --- Mass-Spring-Damper System ---
# m*x'' + c*x' + k*x = F(t)
# State: [position, velocity]
def simulate_msd(kp, ki, kd, setpoint=1.0, dt=0.005, total_time=10.0,
                 mass=1.0, spring=1.0, damping=0.3):
    pid = PIDController(kp, ki, kd)
    t = np.arange(0, total_time, dt)

    x, v = 0.0, 0.0  # position, velocity
    positions = []

    for _ in t:
        F = pid.compute(setpoint, x, dt)
        # Equations of motion
        a = (F - damping * v - spring * x) / mass
        v += a * dt
        x += v * dt
        positions.append(x)

    return t, positions


# --- Initial gains ---
KP0, KI0, KD0 = 15.0, 2.0, 5.0
SETPOINT = 1.0

t, pos = simulate_msd(KP0, KI0, KD0)

# --- Layout ---
fig = plt.figure(figsize=(12, 8))
fig.suptitle('Day 4 — PID on Mass-Spring-Damper System', fontsize=13, fontweight='bold')
gs = GridSpec(2, 1, figure=fig, top=0.90, bottom=0.32, hspace=0.35)

ax_main = fig.add_subplot(gs[0])
ax_zoom = fig.add_subplot(gs[1])

line_main, = ax_main.plot(t, pos, color='royalblue', linewidth=2, label='Position')
ax_main.axhline(SETPOINT, color='gray', linestyle='--', linewidth=1.2, label='Setpoint')
ax_main.set_ylabel('Position (m)')
ax_main.set_xlabel('Time (s)')
ax_main.set_ylim(-0.5, 2.0)
ax_main.legend(fontsize=8)
ax_main.grid(True, alpha=0.4)
ax_main.set_title('Full Response', fontsize=10)

line_zoom, = ax_zoom.plot(t, pos, color='seagreen', linewidth=2, label='Position (zoomed)')
ax_zoom.axhline(SETPOINT, color='gray', linestyle='--', linewidth=1.2)
ax_zoom.set_ylabel('Position (m)')
ax_zoom.set_xlabel('Time (s)')
ax_zoom.set_ylim(0.8, 1.2)
ax_zoom.grid(True, alpha=0.4)
ax_zoom.set_title('Steady-State Zoom', fontsize=10)

# --- Sliders ---
ax_kp = fig.add_axes([0.15, 0.20, 0.7, 0.03])
ax_ki = fig.add_axes([0.15, 0.14, 0.7, 0.03])
ax_kd = fig.add_axes([0.15, 0.08, 0.7, 0.03])

s_kp = Slider(ax_kp, 'Kp', 0.1, 40.0, valinit=KP0, color='royalblue')
s_ki = Slider(ax_ki, 'Ki', 0.0, 10.0, valinit=KI0, color='tomato')
s_kd = Slider(ax_kd, 'Kd', 0.0, 20.0, valinit=KD0, color='seagreen')


def update(val):
    kp = s_kp.val
    ki = s_ki.val
    kd = s_kd.val
    _, new_pos = simulate_msd(kp, ki, kd)
    line_main.set_ydata(new_pos)
    line_zoom.set_ydata(new_pos)
    fig.canvas.draw_idle()


s_kp.on_changed(update)
s_ki.on_changed(update)
s_kd.on_changed(update)

plt.show()
