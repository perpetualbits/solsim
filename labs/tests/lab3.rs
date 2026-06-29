//! Lab 3 grader — checks `euler_step` (forward Euler) and shows it drifts.
//!
//! Run with: `cargo test --test lab3`
//! Note: this lab uses your Lab 2 `gravity_acceleration`, so finish Lab 2 first.

use glam::DVec3;
use labs::{euler_step, GM_SUN};

/// One forward-Euler step matches the hand-computed values exactly.
///
/// Start at 1 AU moving sideways at the circular speed. Forward Euler uses the
/// start-of-step values: the position moves by the *old* velocity, and the velocity
/// changes by gravity. We check both against numbers worked out by hand.
#[test]
fn one_step_matches_hand_calculation() {
    let vc = GM_SUN.sqrt(); // circular speed at r = 1 AU
    let r = DVec3::new(1.0, 0.0, 0.0);
    let v = DVec3::new(0.0, vc, 0.0);
    let dt = 0.1;

    let (r1, v1) = euler_step(r, v, GM_SUN, dt);

    // a at the start = -GM_SUN in x; r moves by old v; v changes by a·dt.
    let expected_r = DVec3::new(1.0, 0.1 * vc, 0.0);
    let expected_v = DVec3::new(-0.1 * GM_SUN, vc, 0.0);
    assert!(
        (r1 - expected_r).length() < 1e-15,
        "position: expected {expected_r:?}, got {r1:?}"
    );
    assert!(
        (v1 - expected_v).length() < 1e-15,
        "velocity: expected {expected_v:?}, got {v1:?}"
    );
}

/// Forward Euler slowly *gains* energy, so a circular orbit spirals outward.
///
/// This is the whole point of the lab: the obvious method is not good enough. We
/// start on a perfect circle (radius 1 AU) and step around it; after many steps the
/// radius has clearly grown — a flaw we fix with RK4 in Lab 4.
#[test]
fn euler_drifts_outward() {
    let mut r = DVec3::new(1.0, 0.0, 0.0);
    let mut v = DVec3::new(0.0, GM_SUN.sqrt(), 0.0);
    for _ in 0..600 {
        let (rn, vn) = euler_step(r, v, GM_SUN, 1.0);
        r = rn;
        v = vn;
    }
    assert!(
        r.length() > 1.05,
        "forward Euler should spiral outward (radius grows), got r = {}",
        r.length()
    );
}
