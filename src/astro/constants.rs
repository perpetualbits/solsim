//! Physical and astronomical constants, all in the project's unit system.
//!
//! Units used everywhere in the simulation: **length in astronomical units (AU)**,
//! **time in days**, **angles in radians**. Keeping one consistent set of units
//! means the formulas in later phases stay simple — no hidden conversion factors.

/// Gaussian gravitational constant `k`.
///
/// What: a fixed number that sets the strength of gravity in AU–day units.
/// How/why: instead of using `G` in SI units, astronomers use `k` so that the
/// Sun's gravity comes out as a clean value (see [`GM_SUN`]); this is the value
/// Gauss derived from Earth's orbit.
/// Units: `k` has units of AU^(3/2) · day⁻¹ · (solar mass)^(−1/2); as a number it
/// is dimensionless here because we measure masses in solar masses.
pub const GAUSS_K: f64 = 0.01720209895;

/// Standard gravitational parameter of the Sun, `G·M_sun = k²`.
///
/// What: how strongly the Sun pulls, used directly in the gravity formula.
/// How/why: Newton's law needs the product `G·M`, and in these units that product
/// for the Sun is exactly `k²` — squaring [`GAUSS_K`] gives ≈ 2.959×10⁻⁴.
/// Units: AU³ · day⁻² (because acceleration = GM / r² has units AU·day⁻²).
pub const GM_SUN: f64 = GAUSS_K * GAUSS_K;

/// Speed of light `c` in the project's units.
///
/// What: how far light travels in one day, needed later for the relativity term.
/// How/why: 299 792.458 km/s converted to AU/day equals ≈ 173.144; light is the
/// universe's speed limit and appears in the General-Relativity correction.
/// Units: AU · day⁻¹.
pub const C_LIGHT: f64 = 173.144;

/// Obliquity of the ecliptic `ε` at J2000, in radians.
///
/// What: the tilt between Earth's equator and the plane of its orbit.
/// How/why: 23.4393° expressed in radians; this angle rotates star coordinates
/// from the equatorial system (right ascension/declination) into the ecliptic
/// system we draw in. Earth's axis is tilted by this much, which also causes the
/// seasons.
/// Units: radians.
pub const OBLIQUITY_RAD: f64 = 23.4393 * core::f64::consts::PI / 180.0;

/// Length of one astronomical unit in kilometres.
///
/// What: the conversion factor between AU and km.
/// How/why: one AU is the average Earth–Sun distance; we use it to turn the Moon
/// distance (which the `astro` crate gives in km) into AU.
/// Units: km per AU.
pub const AU_KM: f64 = 149_597_870.7;

/// A single entry in the placeholder body table.
///
/// What: the bare-minimum facts about a body that later phases will need.
/// How/why: for Phase 1 we only store a name and its gravitational parameter; the
/// full catalogue (radii, colours, parents) arrives in Phase 5.
/// Units: `gm` is in AU³ · day⁻² (same units as [`GM_SUN`]).
pub struct Body {
    /// Human-readable name, e.g. "Earth".
    pub name: &'static str,
    /// Gravitational parameter `G·m` of this body, in AU³·day⁻².
    pub gm: f64,
}

/// Placeholder catalogue of the three bodies used in Phase 1.
///
/// What: the Sun, Earth and Moon with their gravitational parameters.
/// How/why: each `G·m` is `k² · (m / M_sun)`, i.e. [`GM_SUN`] scaled by the
/// body's mass in solar masses (mass ratios from the IAU). Later phases replace
/// this with the full solar-system catalogue in `bodies.rs`.
/// Units: `gm` in AU³·day⁻².
pub const BODIES: [Body; 3] = [
    Body {
        name: "Sun",
        gm: GM_SUN,
    },
    Body {
        // Earth's mass is about 1/332946.05 of the Sun's.
        name: "Earth",
        gm: GM_SUN / 332_946.05,
    },
    Body {
        // The Moon's mass is about 1/27068700 of the Sun's.
        name: "Moon",
        gm: GM_SUN / 27_068_700.0,
    },
];
