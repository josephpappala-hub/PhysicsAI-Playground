# PhysicsAI-Playground
Exploring physics + math with Python and AI

Day 1: Implemented a double pendulum simulation in Python with Matplotlib animation

Day 2: Added documentation and comments to double pendulum simulation. Also finished the projectile motion and bonus graphs.

Day 3: Finished writing down the underlying physics to the double pendulum simulation. 

NEW PROJECT - PID

Day 1: Started the PID project today. I've been wanting to do this ever since the double pendulum because that project made me realize I didn't fully understand why the AI was making the control decisions it was — I just knew it worked. PID felt like the right step back to understand the fundamentals of control theory properly.

Spent the first hour just reading. PID stands for Proportional, Integral, Derivative — each term corrects a different part of the error. Proportional reacts to current error, Integral fixes accumulated error over time, Derivative predicts future error based on rate of change. Simple in concept, surprisingly deep in practice.

Built a basic PIDController class and tested it on a first-order system (essentially a heater model: the output decays back naturally and the controller has to push it toward a setpoint). Got a clean plot showing the system converging to 1.0. First run actually worked which I wasn't expecting — usually something breaks immediately.

The plot is basic but satisfying. You can see the output curve climbing and flattening exactly at the setpoint. Tomorrow I want to see what happens when I deliberately use bad gains.

What I learned: The integral term is what eliminates steady-state error. Without it, a pure P controller always undershoots slightly because once the error gets small, the correction force gets small too.
