                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
# Project house rules — read before every change

## What this is
A 3D solar-system simulator in Rust with a Wayland GUI. Educational target:
Dutch **4 VWO** (≈ age 15–16, decent maths/physics, not university). Clarity and
correctness beat cleverness.

## Hard rules
- **Edition** Rust 2021. Keep the project compiling and runnable after every phase.
- **Precision:** all astronomy/physics in `f64` (`glam::DVec3`/`DMat4`). Convert to
  `f32` only at the GPU boundary, using a camera-centred floating origin.
  *Exception:* the galaxy-collision mode's N-body compute
  (`physics::octree`/`particles`/`galaxy_ic`) is `f32` — it is a scale-free *visual*
  sim, not precision astronomy, and f32 halves the tree's memory footprint.
- **Units:** AU, days, radians. `k = 0.01720209895`, `GM_sun = k²`,
  `c = 173.144 AU/day`, obliquity `ε = 23.4393°`. Document every constant where defined.
- **Crate versions:** add `egui-wgpu` first; use the exact `wgpu` version it depends
  on for the custom renderer. Pin all versions in `Cargo.toml` (no `"*"`).
- **No `unwrap()`/`expect()`** outside `main` startup. Handle `Result`s.

## Comment & documentation rule (REQUIRED on every function)
Every function gets a `///` doc-comment block, in **English**, written so a 4 VWO
student understands it. Each block states, in this order:
1. **What** the function computes (one sentence).
2. **How** — the algorithm or formula, written out in plain Unicode maths
   (e.g. `a = G·m / r²`), with a sentence on *why* it works.
3. **The principle/physics** behind it (e.g. "Newton's law of gravitation says…").
4. **Units** of inputs and outputs.
Keep it concrete and short; no jargon left unexplained.

## Tests
Add `#[test]`s for every maths function against a known value:
- JD of 2000-01-01 12:00 TT = 2451545.0
- Earth ≈ 1 AU from Sun
- Mercury GR perihelion advance ≈ 43″/century (closed-form check)
- A known star's B−V → temperature in the right ballpark
Use Meeus "Astronomical Algorithms" example values where available.

## Style
- Small modules per the agreed layout. One responsibility each.
- Prefer pure functions for maths; keep GPU/state separate from physics.
- Update the on-screen controls help and `docs/math-en.md` (and keep its Dutch
  translation `docs/math-nl.md` in sync) whenever behaviour changes.
