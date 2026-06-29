//! Lab 1 grader — checks `advance_position` (straight-line motion).
//!
//! Run with: `cargo test --test lab1`
//! Each test below describes, in plain language, the fact it is checking.

use glam::DVec3;
use labs::advance_position;

/// One step moves the body by exactly velocity × time.
///
/// A body at the origin moving at 1 AU/day in the +x direction should, after half
/// a day, sit at x = 0.5 AU (and nowhere else).
#[test]
fn one_step_moves_by_v_times_dt() {
    let r0 = DVec3::new(0.0, 0.0, 0.0);
    let v = DVec3::new(1.0, 0.0, 0.0); // 1 AU/day along +x
    let r1 = advance_position(r0, v, 0.5);
    assert!(
        (r1 - DVec3::new(0.5, 0.0, 0.0)).length() < 1e-12,
        "after dt=0.5 the body should be at (0.5, 0, 0), got {r1:?}"
    );
}

/// Taking many small steps adds up to the right total distance.
///
/// Four steps of half a day at 1 AU/day must cover 2 AU in a straight line.
#[test]
fn many_steps_add_up() {
    let v = DVec3::new(1.0, 0.0, 0.0);
    let mut r = DVec3::ZERO;
    for _ in 0..4 {
        r = advance_position(r, v, 0.5);
    }
    assert!(
        (r - DVec3::new(2.0, 0.0, 0.0)).length() < 1e-12,
        "four 0.5-day steps should reach (2, 0, 0), got {r:?}"
    );
}

/// Motion works in any direction (it is a vector, not just along one axis).
///
/// A diagonal velocity should move the body diagonally by the same rule.
#[test]
fn works_in_three_dimensions() {
    let r0 = DVec3::new(1.0, 2.0, 3.0);
    let v = DVec3::new(-1.0, 0.5, 2.0);
    let dt = 0.25;
    let expected = r0 + v * dt;
    let got = advance_position(r0, v, dt);
    assert!(
        (got - expected).length() < 1e-12,
        "expected {expected:?}, got {got:?}"
    );
}
