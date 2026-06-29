//! Lab 5 grader — checks `energies` (kinetic + potential) via the virial theorem.
//!
//! Run with: `cargo test --test lab5`

use glam::DVec3;
use labs::{energies, GM_SUN};

/// Kinetic energy is ½·m·v² and is always positive for a moving body.
#[test]
fn kinetic_is_half_m_v_squared() {
    let mass = 3.0e-6; // roughly an Earth mass, in solar masses
    let v = DVec3::new(0.0, 0.02, 0.0);
    let (ke, _pe) = energies(mass, DVec3::new(1.0, 0.0, 0.0), v, GM_SUN);
    let expected = 0.5 * mass * v.length_squared();
    assert!(
        (ke - expected).abs() < 1e-18,
        "KE should be {expected:e}, got {ke:e}"
    );
}

/// Potential energy is negative (gravity is a bound, attractive well).
#[test]
fn potential_is_negative() {
    let (_ke, pe) = energies(3.0e-6, DVec3::new(1.5, 0.0, 0.0), DVec3::ZERO, GM_SUN);
    assert!(pe < 0.0, "potential energy should be negative, got {pe:e}");
}

/// For a circular orbit the virial theorem holds exactly: 2·KE + PE = 0.
///
/// On a circle `v² = GM/r`, which makes the kinetic energy exactly half the size of
/// the (negative) potential energy. This single check grades both halves of the
/// function at once.
#[test]
fn circular_orbit_satisfies_virial() {
    let mass = 3.0e-6;
    let r = DVec3::new(1.0, 0.0, 0.0);
    let v = DVec3::new(0.0, GM_SUN.sqrt(), 0.0); // circular speed
    let (ke, pe) = energies(mass, r, v, GM_SUN);
    assert!(
        (2.0 * ke + pe).abs() < 1e-12 * ke,
        "virial 2·KE+PE should be ~0, got {}",
        2.0 * ke + pe
    );
    assert!(ke + pe < 0.0, "a bound orbit has negative total energy");
}
