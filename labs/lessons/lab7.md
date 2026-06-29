# Lab 7 (stretch) — Mercury, Einstein, and the 43″

**Goal:** reproduce a famous prediction of General Relativity.
**You edit:** `perihelion_advance_arcsec_per_century` in `src/lib.rs`.
**Run:** `cargo test --test lab7`

---

## The story

Newton's gravity says a single planet traces the *same* ellipse forever. But
astronomers measured that Mercury's ellipse slowly **rotates**: its perihelion (the
point closest to the Sun) creeps around by a tiny angle each orbit. Most of it is
caused by the other planets, but a stubborn **43 arc-seconds per century** was left
unexplained — until Einstein's General Relativity predicted exactly that amount. It
was one of the first great confirmations of the theory.

(An arc-second is 1/3600 of a degree — a very small angle. 43″/century is roughly
the width of a coin seen from 100 metres, accumulating over a hundred years.)

## The formula

General Relativity adds a small extra pull. Worked through, it turns each orbit by

```
Δϖ = 6π·G·M / (c²·a·(1 − e²))      (radians per orbit)
```

where `a` is the orbit's size (semi-major axis), `e` its eccentricity, and `c` the
speed of light. To get arc-seconds per century, multiply by the number of orbits in
a century and convert radians to arc-seconds:

```
rate = Δϖ × (36525 / period_days) × (180·3600 / π)
```

## What to do

```rust
use std::f64::consts::PI;
let per_orbit = 6.0 * PI * gm_sun / (C_LIGHT * C_LIGHT * a * (1.0 - e * e));
let orbits_per_century = 36_525.0 / period_days;
let rad_to_arcsec = 180.0 * 3600.0 / PI;
per_orbit * orbits_per_century * rad_to_arcsec
```

(`gm_sun = G·M`, and `C_LIGHT` is provided in the crate.)

## Check yourself

```text
cargo test --test lab7
```

Mercury (`a = 0.387099`, `e = 0.205630`, `period = 87.969` days) should come out
near **43″/century**; Earth, farther out and rounder, gets only a few. You can
*watch* this precession happen in the main app: switch to the GR engine (press `G`)
and turn up its strength with `]` to see the orbit trace a rosette.

## Where this lives in the real app

This is the closed-form check behind `src/physics/forces.rs`; the actual extra pull
(the "1-post-Newtonian" term) is added to the acceleration there and explored
step-by-step in the app's educational mode (press `K`).

🎉 **That is the full proof-of-concept ladder.** You have now written straight-line
motion, gravity, two integrators, energy, Kepler's equation, and a relativistic
prediction — the real heart of a solar-system simulator.
