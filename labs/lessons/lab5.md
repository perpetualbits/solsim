# Lab 5 — Energy, and the virial theorem

**Goal:** compute a body's kinetic and potential energy.
**You edit:** `energies` in `src/lib.rs`. **Run:** `cargo test --test lab5`

---

## The idea

There are two kinds of energy in an orbit:

- **Kinetic** — the energy of motion: `KE = ½·m·v²` (always positive).
- **Potential** — energy stored in gravity: `PE = −G·M·m / r`. It is **negative**
  because you would have to *add* energy to drag the body away from the Sun (out to
  `r → ∞`, where `PE → 0`).

Return them as a pair `(KE, PE)`. In code: `v.length_squared()` is `v²` and
`r.length()` is `r`; `gm_sun` is `G·M`.

## Why it matters

For a closed system the **total** energy `KE + PE` cannot change — it is conserved.
That is a powerful built-in check: if a simulation's total energy drifts, the
numbers are going wrong (you saw Euler do this in Lab 3). The main app draws this
live — press `Y` for the energy graph and watch the total line stay flat.

There is also the elegant **virial theorem**. For a circular orbit the speed
satisfies `v² = G·M/r`, and if you put that into the two formulas you find:

```
KE = ½·m·(G·M/r),   PE = −m·(G·M/r)   ⇒   2·KE + PE = 0
```

So the kinetic energy is exactly half the size of the (negative) potential energy.
The test checks this identity — which only balances if *both* of your formulas are
right.

## Going deeper

The full story — how the equations of motion themselves come from the energy (the
**Hamiltonian**), and a proper derivation of the virial theorem — is written up in
[`../docs/MATHS.md`](../docs/MATHS.md), section 12.

➡️ **Next:** [Lab 6 — finding a planet on its ellipse (Kepler's equation)](lab6.md).
