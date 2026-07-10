# Goal: Polish the simulation into a complete visual tool
# Added: phase portrait (position vs velocity), disturbance pulse at t=5s,
#        performance metrics (rise time, overshoot, settling time), clean dashboard layout

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec


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


def simulate_full(kp, ki, kd, setpoint=1.0, dt=0.005, total_time=10.0,
                  mass=1.0, spring=1.0, damping=0.3, disturbance_time=5.0,
                  disturbance_mag=2.0):
    pid = PIDController(kp, ki, kd)
    t = np.arange(0, total_time, dt)
    x, v = 0.0, 0.0
    positions, velocities, errors, controls = [], [], [], []

    for ti in t:
        # Inject disturbance pulse at disturbance_time
        disturbance = disturbance_mag if abs(ti - disturbance_time) < dt * 2 else 0.0

        u = pid.compute(setpoint, x, dt)
        error = setpoint - x
        a = (u + disturbance - damping * v - spring * x) / mass
        v += a * dt
        x += v * dt

        positions.append(x)
        velocities.append(v)
        errors.append(error)
        controls.append(u)

    return t, positions, velocities, errors, controls


def compute_metrics(t, positions, setpoint=1.0, tolerance=0.02):
    positions = np.array(positions)
    dt = t[1] - t[0]

    # Rise time: first time output crosses 90% of setpoint
    rise_time = None
    for i, p in enumerate(positions):
        if p >= 0.9 * setpoint:
            rise_time = t[i]
            break

    # Overshoot
    peak = np.max(positions[:int(5.0 / dt)])  # only before disturbance
    overshoot = max(0, (peak - setpoint) / setpoint * 100)

    # Settling time: last time error exceeds tolerance (before disturbance)
    settling_time = None
    pre = positions[:int(5.0 / dt)]
    for i in range(len(pre) - 1, -1, -1):
        if abs(pre[i] - setpoint) > tolerance * setpoint:
            settling_time = t[i]
            break

    return rise_time, overshoot, settling_time


# --- Run simulation ---
KP, KI, KD = 15.0, 2.0, 5.0
SETPOINT = 1.0
t, pos, vel, err, ctrl = simulate_full(KP, KI, KD)
rise_t, overshoot, settle_t = compute_metrics(t, pos)

STEP = 8  # animation frame step

# --- Dashboard Layout ---
fig = plt.figure(figsize=(15, 9))
fig.patch.set_facecolor('#0F1117')
fig.suptitle('PID Controller — Visual Dashboard', fontsize=15,
             fontweight='bold', color='white', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38,
                       top=0.93, bottom=0.08, left=0.07, right=0.97)

ax_pos   = fig.add_subplot(gs[0, :2])   # position (wide)
ax_err   = fig.add_subplot(gs[1, :2])   # error
ax_ctrl  = fig.add_subplot(gs[2, :2])   # control signal
ax_phase = fig.add_subplot(gs[:, 2])    # phase portrait (tall)

STYLE = {'facecolor': '#1A1D27'}
for ax in [ax_pos, ax_err, ax_ctrl, ax_phase]:
    ax.set_facecolor('#1A1D27')
    ax.tick_params(colors='#AAAAAA', labelsize=8)
    ax.grid(True, alpha=0.2, color='#444')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

# Position plot
line_pos, = ax_pos.plot([], [], color='#4FA3FF', linewidth=2, label='Position')
ax_pos.axhline(SETPOINT, color='#FF6B6B', linestyle='--', linewidth=1.2, label='Setpoint')
ax_pos.axvline(5.0, color='#FFD700', linestyle=':', linewidth=1, label='Disturbance')
ax_pos.set_xlim(0, t[-1])
ax_pos.set_ylim(-0.3, 2.2)
ax_pos.set_ylabel('Position (m)', color='#CCCCCC', fontsize=9)
ax_pos.set_title('System Response', color='white', fontsize=10, pad=4)
ax_pos.legend(fontsize=7, facecolor='#1A1D27', labelcolor='white', loc='lower right')

# Error plot
line_err, = ax_err.plot([], [], color='#FF6B6B', linewidth=1.6)
ax_err.axhline(0, color='#555', linestyle='--', linewidth=1)
ax_err.axvline(5.0, color='#FFD700', linestyle=':', linewidth=1)
ax_err.set_xlim(0, t[-1])
ax_err.set_ylim(-1.5, 1.2)
ax_err.set_ylabel('Error', color='#CCCCCC', fontsize=9)
ax_err.set_title('Tracking Error', color='white', fontsize=10, pad=4)

# Control signal plot
line_ctrl, = ax_ctrl.plot([], [], color='#50E3C2', linewidth=1.6)
ax_ctrl.axhline(0, color='#555', linestyle='--', linewidth=1)
ax_ctrl.axvline(5.0, color='#FFD700', linestyle=':', linewidth=1)
ax_ctrl.set_xlim(0, t[-1])
ax_ctrl.set_ylim(-5, 25)
ax_ctrl.set_ylabel('Control (u)', color='#CCCCCC', fontsize=9)
ax_ctrl.set_xlabel('Time (s)', color='#CCCCCC', fontsize=9)
ax_ctrl.set_title('Control Signal', color='white', fontsize=10, pad=4)

# Phase portrait
line_phase, = ax_phase.plot([], [], color='#C77DFF', linewidth=1.4, alpha=0.85)
dot_phase,  = ax_phase.plot([], [], 'o', color='white', markersize=6, zorder=5)
ax_phase.axhline(0, color='#555', linestyle='--', linewidth=0.8)
ax_phase.axvline(SETPOINT, color='#FF6B6B', linestyle='--', linewidth=0.8)
ax_phase.set_xlim(-0.3, 2.2)
ax_phase.set_ylim(-3, 3)
ax_phase.set_xlabel('Position (m)', color='#CCCCCC', fontsize=9)
ax_phase.set_ylabel('Velocity (m/s)', color='#CCCCCC', fontsize=9)
ax_phase.set_title('Phase Portrait\n(pos vs vel)', color='white', fontsize=10, pad=4)

# Metrics text box
metrics_text = (
    f"  Kp = {KP}   Ki = {KI}   Kd = {KD}\n"
    f"  Rise Time:     {rise_t:.2f}s\n"
    f"  Overshoot:    {overshoot:.1f}%\n"
    f"  Settle Time:  {settle_t:.2f}s\n"
    f"  Disturbance @ t=5s"
)
fig.text(0.07, 0.01, metrics_text, fontsize=8, color='#AAAAAA',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#1A1D27', edgecolor='#333', alpha=0.9))

frames = range(0, len(t), STEP)


def animate(i):
    line_pos.set_data(t[:i],   pos[:i])
    line_err.set_data(t[:i],   err[:i])
    line_ctrl.set_data(t[:i],  ctrl[:i])
    line_phase.set_data(pos[:i], vel[:i])
    if i > 0:
        dot_phase.set_data([pos[i-1]], [vel[i-1]])
    return line_pos, line_err, line_ctrl, line_phase, dot_phase


ani = animation.FuncAnimation(
    fig, animate, frames=frames,
    interval=16, blit=True, repeat=False
)

plt.show()
