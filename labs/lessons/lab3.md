# Lab 3 — One time step (the simple way), and why it drifts

**Goal:** combine motion (Lab 1) and gravity (Lab 2) into one time step.
**You edit:** `euler_step` in `src/lib.rs`. **Run:** `cargo test --test lab3`
**Needs:** Lab 2 finished first (this calls your `gravity_acceleration`).

---

## The idea

A simulation is just: *work out the pull, change the velocity, move, repeat.* The
simplest way to write one step is called **forward Euler**. Using the values at the
**start** of the step:

```
a     = gravity_acceleration(r, gm)   // the pull right now
r_new = r + v·dt                      // move using the OLD velocity
v_new = v + a·dt                      // then update the velocity
```

Return `(r_new, v_new)`. Run it in a loop and the body orbits.

## The catch

Forward Euler is easy but not very good. Because it assumes the velocity and pull
stay fixed across the whole step, it keeps slightly *overshooting* — and for an
orbit that means it slowly **gains energy** and spirals outward. The second test
deliberately checks this: start on a perfect circle, step many times, and the
radius grows. That is not a bug in your code; it is the method itself being crude.

This is the motivation for Lab 4, where RK4 fixes the drift by looking ahead inside
the step. (You can also *see* energy drift live in the main app: press `Y` for the
energy graph.)

## Think about it

- Why does using the *old* velocity to move tend to overshoot on a curving orbit?
- What happens to the drift if you make `dt` smaller? (Smaller steps help — but a
  better *method* helps far more, for the same amount of work.)

➡️ **Next:** [Lab 4 — a step that does not drift (RK4)](lab4.md).
