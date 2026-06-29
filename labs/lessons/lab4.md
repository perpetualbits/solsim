# Lab 4 — A time step that does not drift (RK4)

**Goal:** take an accurate step so orbits stay closed.
**You edit:** `rk4_step` in `src/lib.rs`. **Run:** `cargo test --test lab4`
**Needs:** Lab 2 finished first.

---

## The idea

Forward Euler (Lab 3) trusted a single straight-line guess for the whole step, and
drifted. **4th-order Runge–Kutta (RK4)** does much better: it samples the motion
**four times** across the step — at the start, two midpoints, and the end — and
blends them with the weights 1-2-2-1, which cancels most of the error.

Our state is the pair `(r, v)`, and its rate of change is `(v, a(r))` — velocity
changes position, acceleration changes velocity. The four samples are:

```
k1r = v             k1v = a(r)
k2r = v + ½dt·k1v   k2v = a(r + ½dt·k1r)
k3r = v + ½dt·k2v   k3v = a(r + ½dt·k2r)
k4r = v + dt·k3v    k4v = a(r + dt·k3r)
```

and the new state is the weighted average:

```
r_new = r + dt/6·(k1r + 2·k2r + 2·k3r + k4r)
v_new = v + dt/6·(k1v + 2·k2v + 2·k3v + k4v)
```

where `a(p)` means `gravity_acceleration(p, gm)`.

## What to do

A tiny closure keeps it readable:

```rust
let a = |p: glam::DVec3| gravity_acceleration(p, gm);
```

Then compute `k1…k4` exactly as above and return `(r_new, v_new)`.

## Check yourself

```text
cargo test --test lab4
```

Two tests: a circular orbit stays round to within a thousandth of an AU over 400
steps (Euler could not do this!), and after one full period the body comes back to
where it started. This is the very method the real `solarsim` uses
(`src/physics/nbody.rs`).

## Think about it

- Compare with Lab 3: same number of *steps*, but RK4 does ~4× the work per step.
  Why is that trade usually worth it? (Hint: halving Euler's error means many more
  steps; RK4's error shrinks far faster as the step gets smaller.)

➡️ **Next:** [Lab 5 — energy, and checking it is conserved](lab5.md).
