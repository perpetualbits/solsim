# Lab 1 — Move a body in a straight line

**Goal:** make a body coast through space at a steady velocity.
**You edit:** `advance_position` in `src/lib.rs`.
**You run:** `cargo test --test lab1`

---

## The idea

Before we add gravity, let's do the simplest possible motion: a body drifting at a
constant velocity, with nothing pushing on it.

You already know this rule from physics:

> distance = speed × time

A simulator does exactly that, but in tiny steps and in 3-D. We keep two things
about the body:

- its **position** `r` — where it is (a point in space), and
- its **velocity** `v` — how fast and in which direction it moves.

After a small time step `dt`, the body has moved by `v·dt` (speed × time, but as a
vector so it keeps its direction). So its new position is:

```
r_new = r + v·dt
```

That is the whole lab. It looks almost too simple — but repeating this line over
and over, while gravity slowly changes `v`, is what makes planets orbit. You are
building the **skeleton** of the simulator.

## Why vectors?

`r` and `v` are `DVec3`: each holds three numbers (x, y, z). Writing `r + v * dt`
adds the step to all three at once, so the body moves correctly in any direction,
not just along one axis. With glam you can multiply a vector by a number
(`v * dt`) and add two vectors (`r + ...`) directly.

## What to do

Open `src/lib.rs`, find `advance_position`, and replace the `todo!(...)` (and the
`let _ = ...` line above it) with the formula. It is a single line.

## Check yourself

```text
cargo test --test lab1
```

You should see three green tests: one step moves by `v·dt`, many steps add up, and
it works in 3-D. Red? Read the message — it tells you what it expected versus what
your code returned.

## Think about it

- If `dt` is huge, is "straight line for the whole step" still a good guess once
  gravity is curving the path? (Keep this question in mind — it comes back in
  Lab 3, when we see a naive method drift.)

➡️ **Next:** [Lab 2 — the pull of gravity](lab2.md), where `v` finally starts to
change.
