//! The forces that drive the integrator: Newtonian gravity + a relativity term.
//!
//! The planets are pulled by the Sun (fixed at the origin) and by each other. On
//! top of that, General Relativity adds a small extra pull that makes orbits
//! slowly turn (precess) — most famously Mercury's 43″ per century.

use glam::DVec3;

use crate::astro::constants::C_LIGHT;

/// Acceleration of every planet from gravity (and optionally the GR term).
///
/// What: returns the acceleration vector for each body in the state.
/// How/why: for each body `i` we sum
/// (1) the Sun's pull `−GM_sun·r⃗ᵢ/|r⃗ᵢ|³` (the Sun sits at the origin),
/// (2) every other planet's pull `Σ GMⱼ·(r⃗ⱼ−r⃗ᵢ)/|r⃗ⱼ−r⃗ᵢ|³`,
/// (3) when `gr_strength > 0`, the 1-post-Newtonian Schwarzschild correction
/// `(μ/(c²r³))·[(4μ/r − v²)·r⃗ + 4(r⃗·v⃗)·v⃗]` with `μ = GM_sun` and `r⃗,v⃗`
/// measured from the Sun.
/// Principle: Newton's law of gravitation gives (1) and (2); the GR term is the
/// leading relativistic correction that explains Mercury's perihelion advance.
/// Units: `pos` in AU, `vel` in AU/day, `gm`/`sun_gm` in AU³·day⁻², `gr_strength`
/// dimensionless; returned accelerations in AU·day⁻².
pub fn accelerations(
    pos: &[DVec3],
    vel: &[DVec3],
    gm: &[f64],
    sun_gm: f64,
    gr_strength: f64,
) -> Vec<DVec3> {
    let n = pos.len();
    let c2 = C_LIGHT * C_LIGHT;
    let mut acc = vec![DVec3::ZERO; n];

    for i in 0..n {
        let ri = pos[i];
        let r = ri.length();
        // (1) Pull of the Sun at the origin.
        let mut a = -sun_gm * ri / (r * r * r);

        // (2) Pull of the other planets.
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = pos[j] - ri;
            let dist = d.length();
            a += gm[j] * d / (dist * dist * dist);
        }

        // (3) General-relativity correction (relative to the Sun).
        if gr_strength != 0.0 {
            let v = vel[i];
            let v2 = v.length_squared();
            let rv = ri.dot(v);
            let mu = sun_gm;
            a += gr_strength * (mu / (c2 * r * r * r)) * ((4.0 * mu / r - v2) * ri + 4.0 * rv * v);
        }

        acc[i] = a;
    }
    acc
}

/// Closed-form GR perihelion advance, in arc-seconds per century.
///
/// What: predicts how fast an orbit's perihelion turns due to relativity.
/// How/why: the standard result per orbit is `Δϖ = 6π·μ / (c²·a·(1−e²))`
/// radians; multiplying by the number of orbits per century and converting
/// radians→arc-seconds gives the famous rate. For Mercury this is ≈43″/century.
/// Principle: this is the textbook check on the 1PN term — independent of our
/// integrator, so it validates the physics constants and formula.
/// Units: `sun_gm` in AU³·day⁻²; `a` in AU; `e` dimensionless; `period_days` in
/// days; returns arc-seconds per (Julian) century.
#[allow(dead_code)] // reference value for the test and the maths manual (Phase 8)
pub fn perihelion_advance_arcsec_per_century(sun_gm: f64, a: f64, e: f64, period_days: f64) -> f64 {
    let per_orbit_rad =
        6.0 * std::f64::consts::PI * sun_gm / (C_LIGHT * C_LIGHT * a * (1.0 - e * e));
    let orbits_per_century = 36_525.0 / period_days;
    let rad_to_arcsec = 180.0 * 3600.0 / std::f64::consts::PI;
    per_orbit_rad * orbits_per_century * rad_to_arcsec
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::constants::GM_SUN;

    /// Mercury's relativistic perihelion advance must come out near 43″/century.
    #[test]
    fn mercury_perihelion_advance() {
        // Mercury: a = 0.387099 AU, e = 0.205630, period = 87.969 days.
        let rate = perihelion_advance_arcsec_per_century(GM_SUN, 0.387099, 0.205630, 87.969);
        assert!(
            (42.0..44.0).contains(&rate),
            "Mercury GR advance = {rate}″/cy"
        );
    }
}
