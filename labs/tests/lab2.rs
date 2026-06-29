//! Lab 2 grader — checks `gravity_acceleration` (Newton's inverse-square pull).
//!
//! Run with: `cargo test --test lab2`

use glam::DVec3;
use labs::{gravity_acceleration, GM_SUN};

/// Gravity points *toward* the Sun (at the origin), never away from it.
///
/// A body out along +x must be pulled back in the −x direction, with no sideways
/// component.
#[test]
fn points_toward_the_sun() {
    let a = gravity_acceleration(DVec3::new(1.0, 0.0, 0.0), GM_SUN);
    assert!(a.x < 0.0, "pull should be in -x (toward the Sun), got {a:?}");
    assert!(
        a.y.abs() < 1e-15 && a.z.abs() < 1e-15,
        "pull should have no sideways part, got {a:?}"
    );
}

/// At 1 AU the strength is exactly G·M (because |a| = G·M / r² and r = 1).
#[test]
fn correct_strength_at_one_au() {
    let a = gravity_acceleration(DVec3::new(1.0, 0.0, 0.0), GM_SUN);
    assert!(
        (a.length() - GM_SUN).abs() < 1e-12 * GM_SUN,
        "|a| at 1 AU should be GM_SUN = {GM_SUN:e}, got {:e}",
        a.length()
    );
}

/// Inverse-square law: twice as far → a quarter of the pull.
#[test]
fn falls_off_as_inverse_square() {
    let near = gravity_acceleration(DVec3::new(1.0, 0.0, 0.0), GM_SUN).length();
    let far = gravity_acceleration(DVec3::new(2.0, 0.0, 0.0), GM_SUN).length();
    assert!(
        (near / far - 4.0).abs() < 1e-9,
        "doubling distance should quarter the pull (ratio 4), got {}",
        near / far
    );
}

/// The direction is exact in 3-D: the pull is anti-parallel to the position.
#[test]
fn anti_parallel_in_three_dimensions() {
    let r = DVec3::new(0.6, -0.5, 0.3);
    let a = gravity_acceleration(r, GM_SUN);
    // a should point exactly opposite to r: the cross product is (near) zero, and
    // the dot product is negative.
    assert!(
        r.cross(a).length() < 1e-15,
        "pull must lie along the Sun line, got {a:?}"
    );
    assert!(r.dot(a) < 0.0, "pull must be toward the Sun, got {a:?}");
}
