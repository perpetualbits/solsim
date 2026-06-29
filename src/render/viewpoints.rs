//! The three camera viewpoints and the maths behind them.
//!
//! - **Ecliptic-North**: high above the Sun on the +z axis, looking straight down,
//!   so the solar system looks like a flat map.
//! - **Free**: the mouse-controlled orbit camera (handled by [`super::camera`]).
//! - **Earth-surface**: standing on the Earth at a chosen place, looking south at
//!   the sky, correctly oriented for the date and time using sidereal time.

use glam::{DMat4, DVec3, Mat4};

use crate::astro::constants::OBLIQUITY_RAD;

/// Which of the three viewpoints is active.
///
/// What: an enum naming the current camera mode.
/// How/why: the `V` key steps through these in the order the manual lists them.
/// Units: none.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Viewpoint {
    EclipticNorth,
    Free,
    EarthSurface,
}

impl Viewpoint {
    /// The next viewpoint in the cycle.
    ///
    /// What: advances Ecliptic-North → Free → Earth-surface → (repeat).
    /// How/why: pressing `V` calls this; a simple match gives a fixed order.
    /// Units: none.
    pub fn next(self) -> Self {
        match self {
            Viewpoint::EclipticNorth => Viewpoint::Free,
            Viewpoint::Free => Viewpoint::EarthSurface,
            Viewpoint::EarthSurface => Viewpoint::EclipticNorth,
        }
    }

    /// A short human-readable name for the HUD.
    ///
    /// What: returns the viewpoint's label.
    /// How/why: shown in the overlay so you know which mode you are in.
    /// Units: none.
    pub fn name(self) -> &'static str {
        match self {
            Viewpoint::EclipticNorth => "Ecliptic-North",
            Viewpoint::Free => "Free",
            Viewpoint::EarthSurface => "Earth-surface",
        }
    }
}

/// Where on Earth the surface viewpoint stands.
///
/// What: an observer's latitude and longitude.
/// How/why: the local sky depends on where you are; this is configurable but
/// defaults to Zutphen in the Netherlands.
/// Units: degrees (latitude north +, longitude east +).
pub struct Observer {
    pub lat_deg: f64,
    pub lon_deg: f64,
}

impl Default for Observer {
    /// Default observer: Zutphen, Netherlands (52.14°N, 6.20°E).
    fn default() -> Self {
        Self {
            lat_deg: 52.14,
            lon_deg: 6.20,
        }
    }
}

/// Field of view, near and far planes for the Earth-surface (sky) view.
const SURFACE_FOVY: f64 = 70.0 * std::f64::consts::PI / 180.0;
const SURFACE_NEAR: f64 = 0.0001;
const SURFACE_FAR: f64 = 10_000.0;

/// Rotate a vector from equatorial to ecliptic coordinates.
///
/// What: tilts a direction by Earth's axial tilt.
/// How/why: equatorial and ecliptic axes differ by the obliquity `ε`; rotating
/// about the shared x-axis converts between them:
/// `x' = x`, `y' = y·cosε + z·sinε`, `z' = −y·sinε + z·cosε`.
/// Principle: Earth's equator is tilted by ε ≈ 23.44° from its orbital plane.
/// Units: input and output are direction vectors (dimensionless); `eps` in radians.
fn equatorial_to_ecliptic(v: DVec3, eps: f64) -> DVec3 {
    let (s, c) = eps.sin_cos();
    DVec3::new(v.x, v.y * c + v.z * s, -v.y * s + v.z * c)
}

/// Compute the observer's local up/north/east directions in the ecliptic frame.
///
/// What: returns the zenith (straight up), local north and local east as unit
/// vectors in the simulation's ecliptic-J2000 frame.
/// How/why: the zenith points to right ascension = Local Sidereal Time and
/// declination = latitude, so in equatorial coordinates it is
/// `(cosφ·cosθ, cosφ·sinθ, sinφ)` with `θ` = LST. We build it there, rotate it
/// into ecliptic coordinates, then get east and north from cross products with the
/// north celestial pole. LST = Greenwich mean sidereal time + east longitude.
/// Principle: as Earth turns, the sky's apparent rotation is captured by sidereal
/// time; this is what makes the surface view point at the real sky.
/// Units: `jd` in days; returns three dimensionless unit vectors (zenith, north,
/// east).
pub fn local_sky_basis(jd: f64, obs: &Observer) -> (DVec3, DVec3, DVec3) {
    let eps = OBLIQUITY_RAD;
    let gmst = astro::time::mn_sidr(jd); // Greenwich mean sidereal time, radians
    let lst = gmst + obs.lon_deg.to_radians();
    let lat = obs.lat_deg.to_radians();

    let (slat, clat) = lat.sin_cos();
    let (slst, clst) = lst.sin_cos();
    let zenith_eq = DVec3::new(clat * clst, clat * slst, slat);

    let zenith = equatorial_to_ecliptic(zenith_eq, eps).normalize();
    let ncp = equatorial_to_ecliptic(DVec3::Z, eps).normalize();
    let east = ncp.cross(zenith).normalize();
    let north = zenith.cross(east).normalize();
    (zenith, north, east)
}

/// Build the horizon line for the Earth-surface view, in true AU coordinates.
///
/// What: a big circle lying in the observer's horizon plane, centred on the Earth.
/// How/why: the horizon is the set of directions at altitude 0, i.e. the plane
/// spanned by local east and north; we sample a ring `earth + R·(cos t·east +
/// sin t·north)` and list it as segment endpoint pairs. Seen from the observer at
/// the Earth's centre, this ring appears exactly as the horizon line.
/// Units: `earth` in AU; `east`/`north` are unit vectors; returns AU positions.
pub fn horizon_segments(earth: DVec3, east: DVec3, north: DVec3) -> Vec<DVec3> {
    let radius = 50.0;
    let steps = 128;
    let mut segments = Vec::new();
    let mut prev = earth + radius * east;
    for i in 1..=steps {
        let t = std::f64::consts::TAU * (i as f64) / (steps as f64);
        let p = earth + radius * (t.cos() * east + t.sin() * north);
        segments.push(prev);
        segments.push(p);
        prev = p;
    }
    segments
}

/// View-projection matrix for the Earth-surface (sky) view.
///
/// What: the matrix for an observer at the Earth looking toward the southern
/// horizon.
/// How/why: the eye sits at the floating-origin centre (the Earth) looking along
/// `forward` with the local zenith as "up"; `forward` is normally due south
/// (`−north`). Bodies then appear at their correct altitude and azimuth.
/// Units: `forward`/`up` are unit vectors; `aspect` is width/height; returns a
/// dimensionless matrix.
pub fn earth_surface_view_proj(forward: DVec3, up: DVec3, aspect: f32) -> Mat4 {
    let view = DMat4::look_at_rh(DVec3::ZERO, forward, up);
    let proj = DMat4::perspective_rh(SURFACE_FOVY, aspect as f64, SURFACE_NEAR, SURFACE_FAR);
    (proj * view).as_mat4()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The local up/north/east directions must be unit-length and mutually
    /// perpendicular (a valid coordinate frame for the observer's sky).
    #[test]
    fn sky_basis_is_orthonormal() {
        let obs = Observer::default();
        let (zenith, north, east) = local_sky_basis(2_451_545.0, &obs);
        for v in [zenith, north, east] {
            assert!((v.length() - 1.0).abs() < 1e-9, "not unit length: {v:?}");
        }
        assert!(zenith.dot(north).abs() < 1e-9);
        assert!(zenith.dot(east).abs() < 1e-9);
        assert!(north.dot(east).abs() < 1e-9);
    }

    /// Rotating equatorial → ecliptic is a pure rotation, so lengths are kept.
    #[test]
    fn eq_to_ecl_preserves_length() {
        let v = DVec3::new(0.3, -0.5, 0.8);
        let r = equatorial_to_ecliptic(v, OBLIQUITY_RAD);
        assert!(
            (r.length() - v.length()).abs() < 1e-12,
            "length changed: {r:?}"
        );
    }
}
