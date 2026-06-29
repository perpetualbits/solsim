// Reference solution for Lab 7 — try it yourself first!
//
// Not compiled; reference only. Replace the body of
// `perihelion_advance_arcsec_per_century` in ../src/lib.rs.

pub fn perihelion_advance_arcsec_per_century(gm_sun: f64, a: f64, e: f64, period_days: f64) -> f64 {
    use std::f64::consts::PI;
    // How much the perihelion turns each orbit (radians).
    let per_orbit_rad = 6.0 * PI * gm_sun / (C_LIGHT * C_LIGHT * a * (1.0 - e * e));
    // Orbits in a Julian century, and radians → arc-seconds.
    let orbits_per_century = 36_525.0 / period_days;
    let rad_to_arcsec = 180.0 * 3600.0 / PI;
    per_orbit_rad * orbits_per_century * rad_to_arcsec
}
