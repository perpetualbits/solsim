# Lab 6 — Where is the planet? (Kepler's equation)

**Goal:** solve an equation that cannot be solved with algebra.
**You edit:** `solve_kepler` in `src/lib.rs`. **Run:** `cargo test --test lab6`

---

## The idea

A planet does not move at a steady angle around its ellipse — it speeds up near the
Sun and slows down far away (Kepler's second law). Astronomers handle this with two
angles:

- the **mean anomaly** `M` — a pretend angle that *does* grow steadily with time
  (like a clock hand), and
- the **eccentric anomaly** `E` — the angle that actually pins down the planet's
  place on the ellipse.

They are tied together by **Kepler's equation**:

```
M = E − e·sin E
```

where `e` is the eccentricity (how stretched the ellipse is). The problem: you are
given `M` and `e` and want `E` — but you cannot rearrange this for `E`, because `E`
is stuck both inside and outside a sine.

## The method: Newton's method

When you cannot solve an equation directly, you *guess and improve*. Write the
equation as "find `E` where `f(E) = E − e·sin E − M = 0`". Newton's method jumps to
where the tangent line hits zero:

```
E ← E − f(E) / f′(E)     with   f′(E) = 1 − e·cos E
```

Start from `E = M` and repeat a handful of times (8 is plenty). Each round roughly
*doubles* the number of correct digits, so it converges almost instantly.

## What to do

```rust
let mut big_e = mean_anomaly;
for _ in 0..8 {
    let f  = big_e - eccentricity * big_e.sin() - mean_anomaly;
    let fp = 1.0 - eccentricity * big_e.cos();
    big_e -= f / fp;
}
big_e
```

## Check yourself

```text
cargo test --test lab6
```

Three tests: with `e = 0` the answer is just `M`; the answer always satisfies
`M = E − e·sin E` to high precision; and it matches a known value from Meeus'
*Astronomical Algorithms* (`M = 5°, e = 0.0167 → E ≈ 5.0855°`). The main app uses
this exact solver for the moons (`src/astro/elements.rs`).

➡️ **Next (stretch):** [Lab 7 — Mercury and Einstein](lab7.md).
