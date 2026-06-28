//! The logarithmic distance transform (a *display-only* warp).
//!
//! The solar system spans a huge range of distances — Mercury at 0.4 AU, Neptune
//! at 30 AU — so in a true-scale view the outer planets are far off-screen. This
//! module squashes distance-from-the-Sun logarithmically so the whole system fits
//! in one view, while keeping each body in its true *direction*. It never touches
//! the stored positions or the physics — it is applied only when drawing.

use glam::DVec3;

/// Distance scale `r₀` in the transform, in AU.
///
/// What: the distance at which the logarithm "turns over" from roughly linear to
/// compressing.
/// How/why: with `r₀ = 1 AU` the inner planets keep sensible spacing while the
/// outer ones are pulled in.
/// Units: AU.
const REF_DISTANCE_AU: f64 = 1.0;

/// Overall display scale `R₀` applied to the compressed radius.
///
/// What: how big the compressed system is in display units.
/// How/why: chosen so Neptune (≈30 AU) lands near one display unit, giving a tidy
/// view you can frame at a glance.
/// Units: display units per natural-log unit.
const DISPLAY_SCALE: f64 = 0.3;

/// Compress a Sun-centred position logarithmically, keeping its direction.
///
/// What: maps a true position to its on-screen display position in log mode.
/// How/why: replace the distance-from-Sun `r = |p|` with
/// `r_disp = R₀·ln(1 + r/r₀)` and keep the same unit direction, so a body stays on
/// the same bearing from the Sun but much closer in. The Sun itself (r = 0) maps
/// to the origin. Because `ln` grows ever more slowly, far-apart outer planets are
/// pulled into view without the inner ones collapsing onto the Sun.
/// Principle: a logarithm turns multiplicative spacing (each planet ~1.5–2× the
/// last) into roughly even spacing — the same trick as a slide rule.
/// Units: input and output in AU / display units respectively (both lengths);
/// directions are preserved.
pub fn compress(p: DVec3) -> DVec3 {
    let r = p.length();
    if r < 1.0e-12 {
        return DVec3::ZERO;
    }
    let r_disp = DISPLAY_SCALE * (1.0 + r / REF_DISTANCE_AU).ln();
    p * (r_disp / r)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Sun (origin) stays at the origin.
    #[test]
    fn sun_stays_at_origin() {
        assert!(compress(DVec3::ZERO).length() < 1e-12);
    }

    /// Direction is unchanged; only the distance shrinks.
    #[test]
    fn keeps_direction_shrinks_distance() {
        let p = DVec3::new(3.0, -4.0, 0.0); // 5 AU away
        let c = compress(p);
        // Same direction.
        assert!(c.normalize().distance(p.normalize()) < 1e-9);
        // Closer in than the true distance.
        assert!(c.length() < p.length());
    }

    /// Farther bodies still map farther out (the order is preserved).
    #[test]
    fn monotonic_in_distance() {
        let earth = compress(DVec3::new(1.0, 0.0, 0.0)).length();
        let jupiter = compress(DVec3::new(5.2, 0.0, 0.0)).length();
        let neptune = compress(DVec3::new(30.0, 0.0, 0.0)).length();
        assert!(earth < jupiter && jupiter < neptune);
    }
}
