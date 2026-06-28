//! Where the Sun, Earth and Moon are, at a given moment.
//!
//! An **ephemeris** is a table (or here, a formula) that tells you a body's
//! position for any date. We work in the **heliocentric ecliptic J2000** frame:
//! the Sun sits at the origin, the x–y plane is Earth's orbital plane as it was on
//! 1 January 2000, and distances are in AU. Every position is a 3-D vector.

use glam::DVec3;

use super::constants::AU_KM;

/// Position of the Sun.
///
/// What: returns the Sun's location, which is always the origin.
/// How/why: we *chose* a Sun-centred frame, so by definition the Sun is at
/// (0, 0, 0); having a function for it keeps the three bodies symmetric in code.
/// Units: input JD in days (unused); output is a position in AU.
pub fn sun_position(_jd: f64) -> DVec3 {
    DVec3::ZERO
}

/// The eight planets, used to pick a VSOP87 series.
///
/// What: a name for each planet so callers can ask for its position.
/// How/why: lets one function ([`planet_position`]) cover all eight by matching on
/// this, instead of eight separate calls.
/// Units: none.
#[derive(Clone, Copy)]
pub enum Planet {
    Mercury,
    Venus,
    Earth,
    Mars,
    Jupiter,
    Saturn,
    Uranus,
    Neptune,
}

/// Position of any of the eight planets at a given Julian Date.
///
/// What: returns a planet's heliocentric position.
/// How/why: VSOP87A gives heliocentric ecliptic rectangular coordinates (AU) in
/// the J2000 frame for each planet by summing many `A·cos(B + C·T)` terms; we just
/// pick the right series for the requested planet and copy the result into a
/// vector. Same theory and frame as [`earth_position`], so all planets share one
/// consistent coordinate system.
/// Principle: each planet follows a slowly-shifting ellipse; VSOP87 captures those
/// shifts caused by the planets tugging on one another.
/// Units: input JD in days; output a position in AU.
pub fn planet_position(planet: Planet, jd: f64) -> DVec3 {
    let p = match planet {
        Planet::Mercury => vsop87::vsop87a::mercury(jd),
        Planet::Venus => vsop87::vsop87a::venus(jd),
        Planet::Earth => vsop87::vsop87a::earth(jd),
        Planet::Mars => vsop87::vsop87a::mars(jd),
        Planet::Jupiter => vsop87::vsop87a::jupiter(jd),
        Planet::Saturn => vsop87::vsop87a::saturn(jd),
        Planet::Uranus => vsop87::vsop87a::uranus(jd),
        Planet::Neptune => vsop87::vsop87a::neptune(jd),
    };
    DVec3::new(p.x, p.y, p.z)
}

/// Position of the Earth at a given Julian Date.
///
/// What: returns Earth's heliocentric position.
/// How/why: the VSOP87A theory sums hundreds of small periodic terms of the form
/// `A·cos(B + C·T)` to reproduce Earth's orbit very accurately; the `vsop87`
/// crate does that sum for us and returns rectangular AU coordinates already in
/// the ecliptic-J2000 frame, so we copy them straight into a vector.
/// Principle: planets follow elliptical orbits (Kepler), but those ellipses slowly
/// shift due to the other planets — VSOP87 captures all of that.
/// Units: input JD in days; output is a position in AU.
pub fn earth_position(jd: f64) -> DVec3 {
    let p = vsop87::vsop87a::earth(jd);
    DVec3::new(p.x, p.y, p.z)
}

/// Position of the Moon at a given Julian Date.
///
/// What: returns the Moon's heliocentric position.
/// How/why: the `astro` crate's lunar theory (a partial ELP-2000/82 model) gives
/// the Moon's position *relative to the Earth* as an ecliptic longitude λ,
/// latitude β and distance r (in km). We turn those into rectangular coordinates
/// with `x = r·cosβ·cosλ`, `y = r·cosβ·sinλ`, `z = r·sinβ`, convert km → AU, and
/// add Earth's heliocentric position to place the Moon in the Sun-centred frame.
/// Principle: positions add like vectors — Sun→Moon = Sun→Earth + Earth→Moon.
/// (The lunar data is referred to the equinox of date rather than J2000; the tiny
/// resulting angle error is harmless for our purposes.)
/// Units: input JD in days; output is a position in AU.
pub fn moon_position(jd: f64) -> DVec3 {
    let (ecl, dist_km) = astro::lunar::geocent_ecl_pos(jd);
    let r_au = dist_km / AU_KM;
    let geocentric = DVec3::new(
        r_au * ecl.lat.cos() * ecl.long.cos(),
        r_au * ecl.lat.cos() * ecl.long.sin(),
        r_au * ecl.lat.sin(),
    );
    earth_position(jd) + geocentric
}

/// Estimate a body's velocity from its position function.
///
/// What: returns how fast and in which direction a body is moving.
/// How/why: we cannot read velocity directly, so we sample the position a little
/// before and a little after the moment and divide the change by the time gap:
/// `v ≈ (r(t+δ) − r(t−δ)) / (2δ)`. This "central difference" cancels most of the
/// error and is more accurate than looking only forwards.
/// Principle: velocity is the rate of change of position — exactly what a
/// difference quotient approximates, and the basis for seeding the physics engine
/// in Phase 7.
/// Units: `jd` and `delta` in days; output is a velocity in AU · day⁻¹.
pub fn velocity_fd(position: impl Fn(f64) -> DVec3, jd: f64, delta: f64) -> DVec3 {
    (position(jd + delta) - position(jd - delta)) / (2.0 * delta)
}

#[cfg(test)]
mod tests {
    use super::super::time::J2000;
    use super::*;

    /// At J2000 the Earth should be about 1 AU from the Sun.
    #[test]
    fn earth_is_about_one_au() {
        let d = (earth_position(J2000) - sun_position(J2000)).length();
        assert!(
            (0.98..=1.02).contains(&d),
            "Earth-Sun distance was {d} AU"
        );
    }

    /// The Moon should be a few ten-thousandths of an AU from the Earth.
    #[test]
    fn moon_is_near_earth() {
        let d = (moon_position(J2000) - earth_position(J2000)).length();
        assert!(
            (0.0024..=0.0027).contains(&d),
            "Earth-Moon distance was {d} AU"
        );
    }

    /// A finite-difference velocity of straight-line motion recovers the slope.
    #[test]
    fn velocity_of_linear_motion() {
        // A body whose position grows by (2, -1, 0.5) AU per day, measured a few
        // days from the reference (small numbers keep round-off negligible).
        let pos = |t: f64| DVec3::new(2.0 * t, -1.0 * t, 0.5 * t);
        let v = velocity_fd(pos, 10.0, 0.01);
        assert!(
            (v - DVec3::new(2.0, -1.0, 0.5)).length() < 1e-9,
            "velocity was {v:?}"
        );
    }
}
