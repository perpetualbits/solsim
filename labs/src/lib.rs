//! # Build-your-own solar-system simulator — the labs
//!
//! Welcome! In these labs **you** write the physics that makes a planet orbit the
//! Sun. Each lab asks you to fill in one small function. A test then checks your
//! answer against a value we already know is correct — so the computer tells you
//! the moment you get it right.
//!
//! ## How to work
//! 1. Open the lesson for the lab, e.g. `labs/lessons/lab1.md`.
//! 2. Find the matching function below and replace its `todo!(...)` with real code.
//! 3. Run the test for that lab:
//!    ```text
//!    cargo test --test lab1
//!    ```
//!    Red means "not yet"; green means "correct!". Run `cargo test` to check all.
//! 4. Stuck? A worked answer is in `labs/solutions/` — but try first; the struggle
//!    is where the learning happens.
//!
//! ## The units we use (the same as real astronomers, and the main simulator)
//! * **distance** in astronomical units (**AU**): 1 AU = the average Earth–Sun
//!   distance.
//! * **time** in **days**.
//! * **angles** in **radians**.
//!
//! Keeping one set of units means the formulas have no hidden conversion factors.

use glam::DVec3;

/// Gaussian gravitational constant `k` (sets gravity's strength in AU–day units).
///
/// What: a fixed number that makes the Sun's gravity come out to a clean value.
/// How/why: astronomers use `k` instead of the SI constant `G` so that, with mass
/// measured in solar masses, the Sun's pull is exactly `k²` (see [`GM_SUN`]).
/// Units: as a bare number here (because masses are in solar masses).
pub const GAUSS_K: f64 = 0.01720209895;

/// The Sun's gravitational parameter `G·M_sun = k²`.
///
/// What: how strongly the Sun pulls — the one number the gravity formula needs.
/// How/why: Newton's law uses the product `G·M`, and in these units that product
/// for the Sun is exactly `k²` ≈ 2.959×10⁻⁴.
/// Units: AU³·day⁻² (because acceleration = G·M / r² has units AU·day⁻²).
pub const GM_SUN: f64 = GAUSS_K * GAUSS_K;

/// **Lab 1 — Move a body in a straight line.**
///
/// What: returns where a body is after a small time step `dt`, if nothing pushes
/// it (no gravity yet — just coasting).
/// How/why: in a tiny step the body moves at a steady velocity, so its new place
/// is the old place plus "how far it travelled": `r_new = r + v·dt`. Doing this
/// over and over is the skeleton of *every* simulation — we will add the pull of
/// gravity in Lab 2.
/// Principle: this is just "distance = speed × time", written for vectors so it
/// works in 3-D and keeps the direction of travel.
/// Units: `r` in AU, `v` in AU·day⁻¹, `dt` in days; returns a position in AU.
pub fn advance_position(r: DVec3, v: DVec3, dt: f64) -> DVec3 {
    // Delete the line below once you start; it just silences "unused" warnings.
    let _ = (r, v, dt);
    // TODO (Lab 1): return the new position. See labs/lessons/lab1.md.
    todo!("Lab 1: return r + v·dt")
}

/// **Lab 2 — The pull of gravity (acceleration toward the Sun).**
///
/// What: returns the acceleration vector that the Sun (sitting at the origin)
/// gives a body at position `r`.
/// How/why: Newton's law of gravitation says the pull gets weaker with the square
/// of the distance and points straight at the Sun:
/// `a⃗ = −G·M · r⃗ / |r⃗|³`. The minus sign turns the body's own position vector
/// `r⃗` (which points *away* from the Sun) into a pull pointing *toward* it; the
/// `|r⃗|³` in the bottom is `|r⃗|²` for the inverse-square strength times one more
/// `|r⃗|` that turns `r⃗` into a unit direction.
/// Principle: Newton's law of universal gravitation, `a = G·M / r²` in strength,
/// directed along the line to the Sun.
/// Units: `r` in AU, `gm` (= G·M) in AU³·day⁻²; returns acceleration in AU·day⁻².
pub fn gravity_acceleration(r: DVec3, gm: f64) -> DVec3 {
    // Delete the line below once you start; it just silences "unused" warnings.
    let _ = (r, gm);
    // TODO (Lab 2): return -gm * r / |r|³. See labs/lessons/lab2.md.
    // Hint: `r.length()` gives |r⃗|; you can cube it with `len * len * len`.
    todo!("Lab 2: return -gm * r / r.length()^3")
}
