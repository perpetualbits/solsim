//! Lab 7 grader — checks the relativistic perihelion advance formula.
//!
//! Run with: `cargo test --test lab7`

use labs::{perihelion_advance_arcsec_per_century, GM_SUN};

/// Mercury's perihelion advance must come out near the famous 43″/century.
///
/// Mercury: a = 0.387099 AU, e = 0.205630, period = 87.969 days. This is the value
/// Einstein's theory predicted and observations confirmed — a real triumph of
/// physics, and the headline check on the formula.
#[test]
fn mercury_is_about_43_arcsec_per_century() {
    let rate = perihelion_advance_arcsec_per_century(GM_SUN, 0.387099, 0.205630, 87.969);
    assert!(
        (42.0..44.0).contains(&rate),
        "Mercury's GR advance should be ≈ 43″/century, got {rate}"
    );
}

/// Earth's advance is much smaller (it is farther out and nearly circular).
#[test]
fn earth_advance_is_small() {
    let rate = perihelion_advance_arcsec_per_century(GM_SUN, 1.0, 0.0167, 365.256);
    assert!(
        rate > 0.0 && rate < 5.0,
        "Earth's GR advance should be a few ″/century, got {rate}"
    );
}
