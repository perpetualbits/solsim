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

## The lab ladder (proof-of-concept)

| Lab | You build | Lesson | Checks |
|----:|-----------|--------|--------|
| 1 | Straight-line motion `r ← r + v·dt` | [lab1.md](lessons/lab1.md) | `cargo test --test lab1` |
| 2 | Gravity `a = −G·M·r/\|r\|³` | [lab2.md](lessons/lab2.md) | `cargo test --test lab2` |

*More labs are planned* — combining these into a real time step (Euler), seeing it
drift, fixing it with RK4, then energy/virial and Kepler's equation — each mapped
to a piece of the real `solarsim` code. This is the starting slice.

## How this connects to the real simulator

Everything you write here has a twin in the main project, so your work transfers:

| You wrote (here) | Lives in the real app as |
|---|---|
| `advance_position` | the position update inside the RK4 integrator (`src/physics/nbody.rs`) |
| `gravity_acceleration` | `accelerations()` in `src/physics/forces.rs` |

The deeper maths (RK4, the energy/Hamiltonian, the virial theorem, Kepler's
equation) is written up for you in [`../docs/MATHS.md`](../docs/MATHS.md).
