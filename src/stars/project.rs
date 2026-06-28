//! Placing a star on the sky: from right ascension/declination to a direction.
//!
//! Star positions are given in the *equatorial* system (right ascension α and
//! declination δ), but the rest of the simulation uses the *ecliptic* frame. This
//! module turns (α, δ) into a unit direction vector in the ecliptic frame, which
//! the renderer paints onto a far-away background sphere.

use glam::DVec3;

use crate::astro::constants::OBLIQUITY_RAD;

/// Convert a star's (RA, Dec) to a unit direction in the ecliptic frame.
///
/// What: returns a length-1 vector pointing at the star.
/// How/why: first build the equatorial unit vector
/// `(cosδ·cosα, cosδ·sinα, sinδ)`, then rotate it about the x-axis by the
/// obliquity ε to get ecliptic coordinates:
/// `x' = x`, `y' = y·cosε + z·sinε`, `z' = −y·sinε + z·cosε`.
/// Because stars are effectively infinitely far away, only their *direction*
/// matters — parallax from moving around the solar system is negligible.
/// Principle: the equatorial and ecliptic systems differ only by Earth's axial
/// tilt ε, so one rotation converts between them.
/// Units: `ra_deg`/`dec_deg` in degrees; returns a dimensionless unit vector.
pub fn radec_to_ecliptic(ra_deg: f64, dec_deg: f64) -> DVec3 {
    let a = ra_deg.to_radians();
    let d = dec_deg.to_radians();
    let (sa, ca) = a.sin_cos();
    let (sd, cd) = d.sin_cos();
    let eq = DVec3::new(cd * ca, cd * sa, sd);

    let (se, ce) = OBLIQUITY_RAD.sin_cos();
    DVec3::new(eq.x, eq.y * ce + eq.z * se, -eq.y * se + eq.z * ce)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The direction must always be a unit vector.
    #[test]
    fn direction_is_unit_length() {
        for &(ra, dec) in &[(0.0, 0.0), (101.3, -16.7), (270.0, 66.0), (45.0, -30.0)] {
            let v = radec_to_ecliptic(ra, dec);
            assert!((v.length() - 1.0).abs() < 1e-9, "len = {}", v.length());
        }
    }

    /// The vernal equinox (RA 0, Dec 0) is the shared x-axis of both frames, so it
    /// must map to (1, 0, 0) unchanged.
    #[test]
    fn equinox_maps_to_x_axis() {
        let v = radec_to_ecliptic(0.0, 0.0);
        assert!((v - DVec3::X).length() < 1e-9, "got {v:?}");
    }
}
