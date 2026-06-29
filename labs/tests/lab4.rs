//! Lab 4 grader — checks `rk4_step` keeps a circular orbit round.
//!
//! Run with: `cargo test --test lab4`
//! Uses your Lab 2 `gravity_acceleration`, so finish Lab 2 first.

use glam::DVec3;
use labs::{rk4_step, GM_SUN};

/// RK4 keeps a circular orbit at a (very nearly) constant radius.
///
/// Where forward Euler spiralled outward (Lab 3), RK4 should hold the radius to
/// within a thousandth of an AU over hundreds of steps.
#[test]
fn keeps_circular_orbit_round() {
    let mut r = DVec3::new(1.0, 0.0, 0.0);
    let mut v = DVec3::new(0.0, GM_SUN.sqrt(), 0.0);
    for _ in 0..400 {
        let (rn, vn) = rk4_step(r, v, GM_SUN, 1.0);
        r = rn;
        v = vn;
        assert!(
            (r.length() - 1.0).abs() < 1e-3,
            "radius drifted to {}",
            r.length()
        );
    }
}

/// After one full orbit the planet returns to where it started.
///
/// One period at radius 1 AU is `T = 2π·√(a³/GM)`. Stepping for that long should
/// bring the body back close to its starting point — a strong check that the path
/// itself (not just the radius) is right.
#[test]
fn returns_after_one_period() {
    let start = DVec3::new(1.0, 0.0, 0.0);
    let mut r = start;
    let mut v = DVec3::new(0.0, GM_SUN.sqrt(), 0.0);
    let period = std::f64::consts::TAU * (1.0_f64.powi(3) / GM_SUN).sqrt();
    let steps = 2000;
    let dt = period / steps as f64;
    for _ in 0..steps {
        let (rn, vn) = rk4_step(r, v, GM_SUN, dt);
        r = rn;
        v = vn;
    }
    assert!(
        (r - start).length() < 1e-3,
        "after one period the body should return to {start:?}, got {r:?}"
    );
}
