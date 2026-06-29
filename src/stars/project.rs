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
    equatorial_to_ecliptic(equatorial_unit(ra_deg, dec_deg))
}

/// Build the equatorial unit vector for a right ascension and declination.
///
/// What: a length-1 vector pointing at a sky position in the equatorial frame.
/// How/why: the standard spherical → Cartesian map `(cosδ·cosα, cosδ·sinα, sinδ)`.
/// Principle: turning two angles on a sphere into a direction.
/// Units: `ra_deg`/`dec_deg` in degrees; returns a dimensionless unit vector.
fn equatorial_unit(ra_deg: f64, dec_deg: f64) -> DVec3 {
    let (sa, ca) = ra_deg.to_radians().sin_cos();
    let (sd, cd) = dec_deg.to_radians().sin_cos();
    DVec3::new(cd * ca, cd * sa, sd)
}

/// Rotate an equatorial direction into the ecliptic frame.
///
/// What: applies the single rotation about the shared x-axis by the obliquity ε.
/// How/why: `x' = x`, `y' = y·cosε + z·sinε`, `z' = −y·sinε + z·cosε`.
/// Principle: the equatorial and ecliptic frames differ only by Earth's axial
/// tilt ε, so one rotation converts between them.
/// Units: input and output are dimensionless unit vectors.
fn equatorial_to_ecliptic(eq: DVec3) -> DVec3 {
    let (se, ce) = OBLIQUITY_RAD.sin_cos();
    DVec3::new(eq.x, eq.y * ce + eq.z * se, -eq.y * se + eq.z * ce)
}

/// Convert galactic coordinates (l, b) to a unit direction in the ecliptic frame.
///
/// What: returns a length-1 vector pointing at a galactic sky position, used to
/// place the Milky Way band correctly among the stars.
/// How/why: the galactic frame is tilted ≈63° to the equator. We take the known
/// equatorial directions of its two defining axes — the North Galactic Pole
/// (`b = +90°`) and the Galactic Centre (`l = 0, b = 0`) — make them an exact
/// orthonormal right-handed basis (Gram–Schmidt removes the tiny non-orthogonality
/// left by the rounded catalogue values), turn `(l, b)` into a galactic Cartesian
/// vector `(cosb·cosl, cosb·sinl, sinb)`, map it through that basis into the
/// equatorial frame, then rotate by the obliquity into the ecliptic frame.
/// Principle: changing between two rotated coordinate systems is one rotation; the
/// galactic system is fixed relative to the stars (IAU 1958, J2000 values).
/// Units: `l_deg`/`b_deg` in degrees; returns a dimensionless unit vector.
pub fn galactic_to_ecliptic(l_deg: f64, b_deg: f64) -> DVec3 {
    // Equatorial unit vectors of the galactic axes (IAU 1958, J2000).
    let ngp = equatorial_unit(192.85948, 27.12825); // North Galactic Pole → +z
    let gc = equatorial_unit(266.40499, -28.93617); // Galactic Centre   → +x
    let z = ngp;
    let x = (gc - z * gc.dot(z)).normalize(); // force x ⟂ z (Gram–Schmidt)
    let y = z.cross(x); // right-handed: l grows from +x toward +y
    let (sl, cl) = l_deg.to_radians().sin_cos();
    let (sb, cb) = b_deg.to_radians().sin_cos();
    let eq = x * (cb * cl) + y * (cb * sl) + z * sb;
    equatorial_to_ecliptic(eq)
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

    /// The North Galactic Pole (b = +90°) must land on the NGP's own sky direction,
    /// for any longitude we pass in.
    #[test]
    fn galactic_pole_maps_to_ngp() {
        let from_gal = galactic_to_ecliptic(123.0, 90.0);
        let from_eq = radec_to_ecliptic(192.85948, 27.12825);
        assert!(
            (from_gal - from_eq).length() < 1e-9,
            "pole off: {from_gal:?}"
        );
    }

    /// The Galactic Centre (l = 0, b = 0) must point at Sgr A* (its known RA/Dec),
    /// to within the rounding of the catalogue constants.
    #[test]
    fn galactic_centre_maps_to_sgr_a() {
        let from_gal = galactic_to_ecliptic(0.0, 0.0);
        let from_eq = radec_to_ecliptic(266.40499, -28.93617);
        assert!((from_gal - from_eq).length() < 2e-3, "GC off: {from_gal:?}");
    }

    /// Every galactic direction is a unit vector.
    #[test]
    fn galactic_direction_is_unit_length() {
        for &(l, b) in &[(0.0, 0.0), (90.0, 10.0), (200.0, -30.0), (310.0, 60.0)] {
            let v = galactic_to_ecliptic(l, b);
            assert!((v.length() - 1.0).abs() < 1e-9, "len = {}", v.length());
        }
    }
}
