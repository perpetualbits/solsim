# Labs: build your own solar-system simulator

These labs let you write the **physics core** of a solar-system simulator
yourself, one small function at a time. A test checks each answer against a value
we already know is correct, so you get instant feedback: red means "not yet",
green means "you got it".

This crate is deliberately tiny and has **no graphics** — just the maths — so you
can read all of it. When you finish, the very same formulas run inside the full
[`solarsim`](../) app one directory up, now drawing 22 bodies in 3-D.

## Getting started

You need Rust (install from <https://rustup.rs>). Then, from this `labs/` folder:

```text
cargo test            # run every lab's checks
cargo test --test lab1   # run just Lab 1
```

At the start every lab is unsolved, so the tests are **red on purpose**. Your job
is to make them green.

## How each lab works

1. Read the lesson in [`lessons/`](lessons/).
2. Open [`src/lib.rs`](src/lib.rs) and find the matching function.
3. Replace its `todo!(...)` with your code (the doc-comment above the function is
   the specification — it states the formula, the idea, and the units).
4. Run the lab's test until it is green.
5. Stuck? A worked answer is in [`solutions/`](solutions/) — but try first.

## The lab ladder

| Lab | You build | Lesson | Checks |
|----:|-----------|--------|--------|
| 1 | Straight-line motion `r ← r + v·dt` | [lab1.md](lessons/lab1.md) | `cargo test --test lab1` |
| 2 | Gravity `a = −G·M·r/\|r\|³` | [lab2.md](lessons/lab2.md) | `cargo test --test lab2` |
| 3 | One time step (forward Euler) — and watch it drift | [lab3.md](lessons/lab3.md) | `cargo test --test lab3` |
| 4 | An accurate step (RK4) that does not drift | [lab4.md](lessons/lab4.md) | `cargo test --test lab4` |
| 5 | Kinetic & potential energy (virial theorem) | [lab5.md](lessons/lab5.md) | `cargo test --test lab5` |
| 6 | Kepler's equation via Newton's method | [lab6.md](lessons/lab6.md) | `cargo test --test lab6` |
| 7 | Mercury's relativistic perihelion advance (stretch) | [lab7.md](lessons/lab7.md) | `cargo test --test lab7` |

Labs 3–7 build on Lab 2, so work through them in order. By the end you have written
the real heart of a solar-system simulator.

## How this connects to the real simulator

Everything you write here has a twin in the main project, so your work transfers:

| You wrote (here) | Lives in the real app as |
|---|---|
| `advance_position`, `euler_step` | the position/velocity updates inside the integrator (`src/physics/nbody.rs`) |
| `gravity_acceleration` | `accelerations()` in `src/physics/forces.rs` |
| `rk4_step` | `rk4_step()` in `src/physics/nbody.rs` |
| `energies` | `system_energy()` in `src/physics/energy.rs` (energy graph, key `Y`) |
| `solve_kepler` | the moon ephemeris in `src/astro/elements.rs` |
| `perihelion_advance_arcsec_per_century` | the GR check in `src/physics/forces.rs` |

The deeper maths (RK4, the energy/Hamiltonian, the virial theorem, Kepler's
equation) is written up for you in [`../docs/math-en.md`](../docs/math-en.md)
([Nederlands](../docs/math-nl.md)).
