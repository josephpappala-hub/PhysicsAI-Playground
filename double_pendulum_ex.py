import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os

CONFIG = {
    "g": 9.81,
    "L1": 1.0,
    "L2": 1.0,
    "m1": 1.0,
    "m2": 1.0,
    "dt": 0.02,
    "steps": 2000,
    "sim_steps_vis": 800,
    "train_fraction": 0.9,
    "hidden_sizes": [64, 64],
    "lr": 1e-3,
    "epochs": 500,
    "batch_size": 256,
    "random_seed": 42,
    "save_model_path": "pendulum_model.npz",
    "show_energy": True,
    "compare_plots": True,
}

np.random.seed(CONFIG["random_seed"])

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

class DoublePendulum:
    def __init__(self, L1, L2, m1, m2, g):
        self.L1 = L1
        self.L2 = L2
        self.m1 = m1
        self.m2 = m2
        self.g = g

    def derivatives(self, state):
        theta1, omega1, theta2, omega2 = state
        m1, m2, L1, L2, g = self.m1, self.m2, self.L1, self.L2, self.g
        delta = theta2 - theta1
        den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2
        den2 = (L2 / L1) * den1
        eps = 1e-9
        if abs(den1) < eps:
            den1 = eps * np.sign(den1) if den1 != 0 else eps
        if abs(den2) < eps:
            den2 = eps * np.sign(den2) if den2 != 0 else eps
        a1 = (m2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta)
              + m2 * g * np.sin(theta2) * np.cos(delta)
              + m2 * L2 * omega2**2 * np.sin(delta)
              - (m1 + m2) * g * np.sin(theta1)) / den1
        a2 = (-m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta)
              + (m1 + m2) * g * np.sin(theta1) * np.cos(delta)
              - (m1 + m2) * L1 * omega1**2 * np.sin(delta)
              - (m1 + m2) * g * np.sin(theta2)) / den2
        return np.array([omega1, a1, omega2, a2])

    def rk4_step(self, state, dt):
        k1 = self.derivatives(state)
        k2 = self.derivatives(state + 0.5 * dt * k1)
        k3 = self.derivatives(state + 0.5 * dt * k2)
        k4 = self.derivatives(state + dt * k3)
        next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        next_state[0] = wrap_angle(next_state[0])
        next_state[2] = wrap_angle(next_state[2])
        return next_state

    def energy(self, state):
        theta1, omega1, theta2, omega2 = state
        m1, m2, L1, L2, g = self.m1, self.m2, self.L1, self.L2, self.g
        x1 = L1 * np.sin(theta1)
        y1 = -L1 * np.cos(theta1)
        x2 = x1 + L2 * np.sin(theta2)
        y2 = y1 - L2 * np.cos(theta2)
        v1x = L1 * omega1 * np.cos(theta1)
        v1y = L1 * omega1 * np.sin(theta1)
        v2x = v1x + L2 * omega2 * np.cos(theta2)
        v2y = v1y + L2 * omega2 * np.sin(theta2)
        K = 0.5 * m1 * (v1x**2 + v1y**2) + 0.5 * m2 * (v2x**2 + v2y**2)
        U = m1 * g * y1 + m2 * g * y2
        return K + U

class SimpleMLP:
    def __init__(self, input_size, output_size, hidden_sizes, lr=1e-3, seed=0):
        self.sizes = [input_size] + hidden_sizes + [output_size]
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for i in range(len(self.sizes)-1):
            in_s = self.sizes[i]
            out_s = self.sizes[i+1]
            limit = np.sqrt(6 / (in_s + out_s))
            self.weights.append(rng.uniform(-limit, limit, size=(out_s, in_s)))
            self.biases.append(np.zeros((out_s,)))
        self.lr = lr

    def forward(self, x):
        a = x
        caches = [a]
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W.T + b
            if i < len(self.weights) - 1:
                a = np.tanh(z)
            else:
                a = z
            caches.append(a)
        return a, caches

    def predict(self, x):
        y, _ = self.forward(x)
        return y

    def mse_loss(self, y_pred, y_true):
        diff = y_pred - y_true
        return 0.5 * np.mean(np.sum(diff**2, axis=1))

    def train_epoch(self, X, Y, batch_size):
        n = X.shape[0]
        idx = np.arange(n)
        np.random.shuffle(idx)
        losses = []
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start+batch_size]
            x_batch = X[batch_idx]
            y_batch = Y[batch_idx]
            y_pred, caches = self.forward(x_batch)
            diff = (y_pred - y_batch) / x_batch.shape[0]
            grads_w = [None] * len(self.weights)
            grads_b = [None] * len(self.biases)
            delta = diff
            for i in reversed(range(len(self.weights))):
                a_prev = caches[i]
                grads_w[i] = delta.T @ a_prev
                grads_b[i] = np.sum(delta, axis=0)
                if i > 0:
                    W = self.weights[i]
                    delta = (delta @ W) * (1 - caches[i]**2)
            for i in range(len(self.weights)):
                self.weights[i] -= self.lr * grads_w[i]
                self.biases[i] -= self.lr * grads_b[i]
            losses.append(self.mse_loss(y_pred, y_batch))
        return float(np.mean(losses))

def generate_dataset(p, dt, steps, initial_state=None):
    if initial_state is None:
        th1 = (np.pi/2) * (0.8 + 0.4 * (np.random.rand()-0.5))
        th2 = (np.pi/2) * (0.8 + 0.4 * (np.random.rand()-0.5))
        om1 = 0.0
        om2 = 0.0
        state = np.array([th1, om1, th2, om2])
    else:
        state = initial_state.copy()
    X, Y = [], []
    for _ in range(steps):
        next_state = p.rk4_step(state, dt)
        X.append(state.copy())
        Y.append(next_state.copy())
        state = next_state
    return np.array(X), np.array(Y)

def fit_scaler(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0) + 1e-9
    return mu, sigma

def scale(X, mu, sigma):
    return (X - mu) / sigma

def descale(Xs, mu, sigma):
    return Xs * sigma + mu

def train_model(p, cfg):
    dt = cfg["dt"]
    steps = cfg["steps"]
    X, Y = generate_dataset(p, dt, steps)
    N = X.shape[0]
    n_train = int(N * cfg["train_fraction"])
    idx = np.arange(N)
    np.random.shuffle(idx)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val,   Y_val   = X[val_idx],   Y[val_idx]
    X_mu, X_sigma = fit_scaler(X_train)
    Y_mu, Y_sigma = fit_scaler(Y_train)
    Xtr = scale(X_train, X_mu, X_sigma)
    Ytr = scale(Y_train, Y_mu, Y_sigma)
    Xval = scale(X_val, X_mu, X_sigma)
    Yval = scale(Y_val, Y_mu, Y_sigma)
    model = SimpleMLP(4, 4, cfg["hidden_sizes"], lr=cfg["lr"])
    best_val = float("inf")
    best_params = None
    for epoch in range(1, cfg["epochs"]+1):
        train_loss = model.train_epoch(Xtr, Ytr, cfg["batch_size"])
        val_pred = model.predict(Xval)
        val_loss = model.mse_loss(val_pred, Yval)
        if val_loss < best_val:
            best_val = val_loss
            best_params = [W.copy() for W in model.weights], [b.copy() for b in model.biases]
    model.weights, model.biases = best_params
    return model, (X_mu, X_sigma, Y_mu, Y_sigma)

def rollout_rk4(p, initial_state, dt, steps):
    states = [initial_state.copy()]
    s = initial_state.copy()
    for _ in range(steps):
        s = p.rk4_step(s, dt)
        states.append(s.copy())
    return np.array(states)

def rollout_model(model, initial_state, dt, steps, scalers):
    X_mu, X_sigma, Y_mu, Y_sigma = scalers
    s = initial_state.copy()
    states = [s.copy()]
    for _ in range(steps):
        x_scaled = scale(s.reshape(1, -1), X_mu, X_sigma)
        y_scaled = model.predict(x_scaled)
        s_next = descale(y_scaled, Y_mu, Y_sigma).flatten()
        s_next[0] = wrap_angle(s_next[0])
        s_next[2] = wrap_angle(s_next[2])
        s = s_next
        states.append(s.copy())
    return np.array(states)

def animate_results(p, truth, pred, dt):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, xlim=(-2,2), ylim=(-2,2))
    ax.set_aspect('equal')
    line_true, = ax.plot([], [], lw=2, color='blue')
    line_pred, = ax.plot([], [], lw=2, color='red')
    def init():
        # set initial empty data for blitting
        line_true.set_data([], [])
        line_pred.set_data([], [])
        return line_true, line_pred

    def update(i):
        th1, _, th2, _ = truth[i]
        x1 = p.L1 * np.sin(th1)
        y1 = -p.L1 * np.cos(th1)
        x2 = x1 + p.L2 * np.sin(th2)
        y2 = y1 - p.L2 * np.cos(th2)
        line_true.set_data([0, x1, x2], [0, y1, y2])
        th1p, _, th2p, _ = pred[i]
        x1p = p.L1 * np.sin(th1p)
        y1p = -p.L1 * np.cos(th1p)
        x2p = x1p + p.L2 * np.sin(th2p)
        y2p = y1p - p.L2 * np.cos(th2p)
        line_pred.set_data([0, x1p, x2p], [0, y1p, y2p])
        return line_true, line_pred
    # keep a reference to the animation object so it isn't garbage-collected
    ani = animation.FuncAnimation(fig, update, frames=len(truth), interval=dt*1000, blit=True, init_func=init)
    plt.show()
    return ani

if __name__ == "__main__":
    cfg = CONFIG
    p = DoublePendulum(cfg["L1"], cfg["L2"], cfg["m1"], cfg["m2"], cfg["g"])
    model, scalers = train_model(p, cfg)
    init = np.array([np.pi*0.8, 0.0, np.pi*0.3, 0.0])
    truth = rollout_rk4(p, init, cfg["dt"], cfg["sim_steps_vis"])
    pred = rollout_model(model, init, cfg["dt"], cfg["sim_steps_vis"], scalers)
    # keep the returned animation object in scope so it is not garbage-collected
    ani = animate_results(p, truth, pred, cfg["dt"])
