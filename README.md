# PhysicsAI-Playground
Exploring physics + math with Python and AI

Day 1: Implemented a double pendulum simulation in Python with Matplotlib animation

Day 2: Added documentation and comments to double pendulum simulation. Also finished the projectile motion and bonus graphs.

Day 3: Finished writing down the underlying physics to the double pendulum simulation. 

NEW PROJECT - PID

Day 1: Started the PID project today. I've been wanting to do this ever since the double pendulum because that project made me realize I didn't fully understand why the AI was making the control decisions it was, I just knew it worked. PID felt like the right step back to understand the fundamentals of control theory properly.

Spent the first hour just reading. PID stands for Proportional, Integral, Derivative — each term corrects a different part of the error. Proportional reacts to current error, Integral fixes accumulated error over time, Derivative predicts future error based on rate of change. Simple in concept, surprisingly deep in practice.

Built a basic PIDController class and tested it on a first-order system (essentially a heater model: the output decays back naturally and the controller has to push it toward a setpoint). Got a clean plot showing the system converging to 1.0. First run actually worked which I wasn't expecting, usually something breaks immediately.

The plot is basic but satisfying. You can see the output curve climbing and flattening exactly at the setpoint. Tomorrow I want to see what happens when I deliberately use bad gains.

What I learned: The integral term is what eliminates steady-state error. Without it, a pure P controller always undershoots slightly because once the error gets small, the correction force gets small too.


Day 2 — Breaking Things on Purpose


Today I intentionally used bad PID gains to see what goes wrong. This was more useful than I expected.

Set up four configurations: high Kp (too aggressive), low Kp (too sluggish), no Kd (oscillates), and a well-tuned version. Plotted them in a 2x2 grid so I could compare directly. The high Kp case overshoots badly and rings back and forth before settling. The low Kp case barely moves — it crawls toward the setpoint so slowly it looks broken. The no-Kd case oscillates around the setpoint without ever fully calming down.

The well-tuned case is genuinely satisfying to look at, smooth curve, minimal overshoot, clean settle. It makes sense now why tuning matters so much in real engineering. A poorly tuned PID on an actual system (like a drone or a manufacturing arm) could cause real damage.

Realized I need to start thinking about what system the PID is controlling. Right now it's an abstract first-order model. I want to replace that with something physical. Mass-spring-damper feels like the right next step, it's a real mechanical system that shows up everywhere in engineering.

What I learned: Kd acts like a brake. Without it the system overshoots because it has no way to anticipate that it's approaching the target too fast. Adding Kd damps the response and prevents oscillation.

Day 3 — Making It Move


Added animation today. This took longer than expected, matplotlib's FuncAnimation is a bit finicky with blitting and the frame update logic. Spent about 40 minutes debugging why the animation was freezing before realizing I was re-simulating inside the animation loop instead of pre-computing all the data and just indexing into it. Fixed that and it runs smoothly now.

The visualization now shows three subplots animating simultaneously: system output, error signal, and control signal. Watching all three together is genuinely more
informative than a static plot, you can see the control signal spike hard at the start (the controller is working its hardest), then gradually reduce as the error shrinks. The error plot crosses zero briefly (that's the overshoot) then settles back.

I also slowed the animation down by skipping every 5th frame so it doesn't blur past too fast on screen.

Day 4 — Real Physics + Interactive Sliders

Date: Thursday

This was the biggest day so far. Two major changes: replaced the abstract system with a real mass-spring-damper model, and added interactive sliders so you can retune the PID live without rerunning the script.

The mass-spring-damper is governed by: mx'' + cx' + k*x = F(t), where F is the control force from the PID. This is a second-order system so I had to track both
position and velocity as state variables and integrate them separately. Took a few attempts to get the numerical integration stable, had to reduce dt to 0.005 to stop it from blowing up at high gains.

The sliders use matplotlib.widgets.Slider and call an update() function every time a value changes, which re-simulates and redraws the lines. Dragging the Kp slider up in real time and watching the response get more aggressive is really satisfying. It makes the relationship between gain and behaviour immediately intuitive.

Added a zoomed subplot showing just the steady-state region (y: 0.8 to 1.2) so you can clearly see whether there's any remaining steady-state error.

This version feels like something actually useful. If I were tutoring someone in control systems, I'd open this and let them drag the sliders themselves.

What I learned: Second-order systems are fundamentally richer than first-order ones. You get real oscillation, real overshoot, real settling behaviour. The mass-spring-damper is the "hello world" of mechanical engineering for a reason. Everything from car suspensions to building structures to circuit components maps onto it.

The project is starting to feel like a real teaching tool rather than just a personal exercise. The animation would be useful for actually explaining PID to someone who had never seen it before.

What I learned: Separating simulation from visualization is important. Pre-compute everything, then animate from stored data. Trying to simulate and animate in real time at this fidelity doesn't work well in matplotlib.
