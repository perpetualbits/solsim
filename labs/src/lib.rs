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

/// Speed of light `c`, in the project's units (used by the relativity lab).
///
/// What: how far light travels in one day.
/// How/why: 299 792.458 km/s converted to AU/day ≈ 173.144; light is the
/// universe's speed limit and appears in the General-Relativity correction.
/// Units: AU·day⁻¹.
pub const C_LIGHT: f64 = 173.144;

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

/// **Lab 3 — One time step the simple way (forward Euler).**
///
/// What: advances the body by one step `dt`, returning its new `(position,
/// velocity)`, using gravity from [`gravity_acceleration`] (so finish Lab 2 first).
/// How/why: combine Lab 1 and Lab 2. The "forward Euler" recipe uses the values at
/// the *start* of the step for everything:
/// `a = gravity(r);  r_new = r + v·dt;  v_new = v + a·dt`.
/// (Note `r_new` uses the *old* `v`.) It is the most obvious stepping rule — and,
/// as the lesson shows, it slowly *gains energy*, so the orbit spirals outward.
/// That flaw is exactly why Lab 4 introduces a better method.
/// Principle: Newton's second law as a recipe — acceleration changes velocity,
/// velocity changes position — applied in small straight-line jumps.
/// Units: `r` in AU, `v` in AU·day⁻¹, `gm` in AU³·day⁻², `dt` in days.
pub fn euler_step(r: DVec3, v: DVec3, gm: f64, dt: f64) -> (DVec3, DVec3) {
    // Delete the line below once you start; it just silences "unused" warnings.
    let _ = (r, v, gm, dt);
    // TODO (Lab 3): forward Euler. See labs/lessons/lab3.md.
    todo!("Lab 3: return (r + v·dt, v + a·dt) with a = gravity_acceleration(r, gm)")
}

/// **Lab 4 — One accurate time step (4th-order Runge–Kutta, RK4).**
///
/// What: advances the body by one step `dt` far more accurately than Euler, again
/// returning the new `(position, velocity)` (needs Lab 2).
/// How/why: instead of one estimate of the motion, RK4 takes four — at the start,
/// two midpoints and the end — and averages them with weights 1-2-2-1. The state
/// is `y = (r, v)` and its rate of change is `(v, a(r))`:
/// ```text
/// k1r = v;            k1v = a(r)
/// k2r = v + ½dt·k1v;  k2v = a(r + ½dt·k1r)
/// k3r = v + ½dt·k2v;  k3v = a(r + ½dt·k2r)
/// k4r = v + dt·k3v;   k4v = a(r + dt·k3r)
/// r_new = r + dt/6·(k1r + 2k2r + 2k3r + k4r)
/// v_new = v + dt/6·(k1v + 2k2v + 2k3v + k4v)
/// ```
/// Averaging several tangent estimates cancels the low-order error, so the orbit
/// stays closed where Euler drifts. This is the method the real simulator uses.
/// Principle: follow the true curved path `r¨ = a(r)` by sampling its slope several
/// times per step rather than trusting a single straight-line guess.
/// Units: `r` in AU, `v` in AU·day⁻¹, `gm` in AU³·day⁻², `dt` in days.
pub fn rk4_step(r: DVec3, v: DVec3, gm: f64, dt: f64) -> (DVec3, DVec3) {
    // Delete the line below once you start; it just silences "unused" warnings.
    let _ = (r, v, gm, dt);
    // TODO (Lab 4): RK4 with the four estimates above. See labs/lessons/lab4.md.
    // Tip: a small closure `let a = |p| gravity_acceleration(p, gm);` keeps it tidy.
    todo!("Lab 4: combine the four RK4 estimates")
}

/// **Lab 5 — Kinetic and potential energy.**
///
/// What: returns `(kinetic, potential)` energy of a body orbiting the Sun.
/// How/why:
/// • kinetic (energy of motion):  `KE = ½·m·v²`;
/// • potential (stored in gravity): `PE = −G·M·m / r`, negative because you must
///   add energy to pull the body away from the Sun.
/// Their sum is the *total* energy, which should stay constant as the body orbits —
/// a built-in check on the simulation (see the energy graph, key `Y`, in the app).
/// Principle: conservation of energy; together with the virial theorem
/// (`2·KE + PE = 0` for a circular orbit) this is one of the deepest ideas in
/// mechanics — see `../docs/math-en.md`.
/// Units: `mass` in solar masses, `r` in AU, `v` in AU·day⁻¹, `gm_sun` (= G·M) in
/// AU³·day⁻²; energies in M_sun·AU²·day⁻².
pub fn energies(mass: f64, r: DVec3, v: DVec3, gm_sun: f64) -> (f64, f64) {
    // Delete the line below once you start; it just silences "unused" warnings.
    let _ = (mass, r, v, gm_sun);
    // TODO (Lab 5): return (½·m·v², −G·M·m/r). See labs/lessons/lab5.md.
    // Hint: `v.length_squared()` is v²; `r.length()` is r.
    todo!("Lab 5: return (kinetic, potential)")
}

/// **Lab 6 — Solve Kepler's equation (where is a planet on its ellipse?).**
///
/// What: given the "mean anomaly" `M` (a clock-like angle that grows steadily in
/// time) and the orbit's eccentricity `e`, returns the "eccentric anomaly" `E`, the
/// angle that actually tells you the planet's place on its ellipse.
/// How/why: the two are linked by Kepler's equation `M = E − e·sin E`, which cannot
/// be rearranged for `E` with algebra. We solve it with **Newton's method**: start
/// with a guess `E = M`, then repeatedly improve it with
/// `E ← E − (E − e·sin E − M) / (1 − e·cos E)`. A handful of rounds is plenty,
/// because each round roughly doubles the number of correct digits.
/// Principle: Kepler's second law (a planet sweeps equal areas in equal times)
/// makes `M` advance evenly; converting `M → E` is the step that turns "how much
/// time has passed" into "where the planet is".
/// Units: `mean_anomaly` in radians, `eccentricity` dimensionless (0 ≤ e < 1);
/// returns `E` in radians.
pub fn solve_kepler(mean_anomaly: f64, eccentricity: f64) -> f64 {
    // Delete the line below once you start; it just silences "unused" warnings.
    let _ = (mean_anomaly, eccentricity);
    // TODO (Lab 6): Newton's method on M = E − e·sin E. See labs/lessons/lab6.md.
    todo!("Lab 6: return E solving M = E - e·sin E")
}

/// **Lab 7 (stretch) — Mercury's relativistic perihelion advance.**
///
/// What: returns how fast an orbit's perihelion (closest point to the Sun) slowly
/// turns because of General Relativity, in arc-seconds per century.
/// How/why: Einstein's correction makes each orbit fail to close by a tiny angle
/// `Δϖ = 6π·G·M / (c²·a·(1 − e²))` radians. Multiply by the number of orbits in a
/// century (`36525 / period`) and convert radians → arc-seconds
/// (`× 180·3600 / π`). For Mercury this comes out near the famous **43″/century**,
/// one of the first confirmations of relativity.
/// Principle: the leading (1-post-Newtonian) relativistic term adds a small extra
/// pull; over many orbits it accumulates into a measurable rotation of the ellipse.
/// Units: `gm_sun` (= G·M) in AU³·day⁻², `a` in AU, `e` dimensionless,
/// `period_days` in days; returns arc-seconds per Julian century.
pub fn perihelion_advance_arcsec_per_century(gm_sun: f64, a: f64, e: f64, period_days: f64) -> f64 {
    // Delete the line below once you start; it just silences "unused" warnings.
    let _ = (gm_sun, a, e, period_days);
    // TODO (Lab 7): combine the three factors above. See labs/lessons/lab7.md.
    // Hint: std::f64::consts::PI is π.
    todo!("Lab 7: return the advance in arc-seconds per century")
}
