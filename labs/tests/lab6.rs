//! Lab 6 grader — checks `solve_kepler` (Newton's method on Kepler's equation).
//!
//! Run with: `cargo test --test lab6`

use labs::solve_kepler;

/// With a circular orbit (e = 0) the equation is just M = E, so E must equal M.
#[test]
fn circular_orbit_returns_mean_anomaly() {
    for &m in &[0.0, 0.3, 1.0, 2.5, 6.0] {
        let e = solve_kepler(m, 0.0);
        assert!((e - m).abs() < 1e-12, "e=0: E should equal M={m}, got {e}");
    }
}

/// The returned E must actually satisfy Kepler's equation M = E − e·sin E.
///
/// We plug the answer back in for several realistic eccentricities and check the
/// equation balances to high precision — the definition of a correct solution.
#[test]
fn solution_satisfies_keplers_equation() {
    let cases = [(0.5, 0.2), (2.0, 0.1), (1.0, 0.6), (4.5, 0.3), (3.1, 0.9)];
    for &(m, e) in &cases {
        let big_e = solve_kepler(m, e);
        let residual = big_e - e * big_e.sin() - m;
        assert!(
            residual.abs() < 1e-10,
            "M={m}, e={e}: E={big_e} does not satisfy the equation (residual {residual:e})"
        );
    }
}

/// A known value (Meeus, *Astronomical Algorithms*): M = 5°, e = 0.0167 → E ≈ 5.085°.
#[test]
fn matches_known_meeus_value() {
    let deg = std::f64::consts::PI / 180.0;
    let e = solve_kepler(5.0 * deg, 0.016_71);
    assert!(
        (e / deg - 5.085_5).abs() < 1e-3,
        "expected E ≈ 5.0855°, got {}°",
        e / deg
    );
}
