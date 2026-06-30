//! A small coherent-noise toolkit: value noise and fractional Brownian motion.
//!
//! "Coherent" noise is *smooth* random — nearby points get similar values, with one
//! characteristic blob size — unlike white (TV-static) noise. Summing several
//! octaves of it (fBm) gives the self-similar, cloud- and dust-like fields used both
//! by the procedural clouds (`render::clouds`) and the Milky Way's dust lanes
//! (`stars::galaxy`). Sampling on 3-D points (e.g. a direction on a sphere) keeps it
//! seamless — no edges, no pinching at the poles.

use glam::DVec3;

/// Fractional Brownian motion: sum several octaves of value noise.
///
/// What: returns a smooth fractal value in 0..1 for a 3-D point.
/// How/why: `fbm = Σ amplitudeᵢ·noise(2ⁱ·p)` with the amplitude halving each octave;
/// dividing by the total amplitude keeps the result in 0..1. The doubling frequency
/// with halving strength is what makes it look the same at every scale.
/// Principle: self-similar detail — the heart of fractal textures.
/// Units: `p` dimensionless; returns a value in 0..1.
pub fn fbm(mut p: DVec3, octaves: u32) -> f64 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut norm = 0.0;
    for _ in 0..octaves {
        sum += amp * value_noise(p);
        norm += amp;
        p *= 2.0;
        amp *= 0.5;
    }
    sum / norm
}

/// Smooth 3-D value noise on an integer lattice.
///
/// What: a coherent (smooth) random field in 0..1 — nearby points are similar.
/// How/why: hash the eight surrounding lattice corners to random values, then
/// trilinearly interpolate using a smoothstep "fade" so there are no creases.
/// Principle: interpolating lattice noise gives a band-limited field with one
/// characteristic blob size — the building block stacked by [`fbm`].
/// Units: `p` dimensionless; returns a value in 0..1.
pub fn value_noise(p: DVec3) -> f64 {
    let pf = p.floor();
    let (ix, iy, iz) = (pf.x as i32, pf.y as i32, pf.z as i32);
    let f = p - pf;
    // Smoothstep fade per axis: 3f² − 2f³.
    let fade = |t: f64| t * t * (3.0 - 2.0 * t);
    let (ux, uy, uz) = (fade(f.x), fade(f.y), fade(f.z));

    let c = |dx: i32, dy: i32, dz: i32| hash3(ix + dx, iy + dy, iz + dz);
    // Trilinear blend of the eight corners.
    let x00 = lerp(c(0, 0, 0), c(1, 0, 0), ux);
    let x10 = lerp(c(0, 1, 0), c(1, 1, 0), ux);
    let x01 = lerp(c(0, 0, 1), c(1, 0, 1), ux);
    let x11 = lerp(c(0, 1, 1), c(1, 1, 1), ux);
    let y0 = lerp(x00, x10, uy);
    let y1 = lerp(x01, x11, uy);
    lerp(y0, y1, uz)
}

/// Hash three integer lattice coordinates to a pseudo-random value in 0..1.
///
/// What: a repeatable "random" number for a grid corner.
/// How/why: mix the coordinates with large odd constants and a few xor-shift /
/// multiply rounds so neighbouring corners get unrelated values; divide by the
/// `u32` range to land in 0..1.
/// Principle: a good integer hash behaves like white noise on the lattice.
/// Units: integer inputs; returns a value in 0..1.
fn hash3(x: i32, y: i32, z: i32) -> f64 {
    let mut h = (x as u32).wrapping_mul(0x8DA6_B343)
        ^ (y as u32).wrapping_mul(0xD816_3841)
        ^ (z as u32).wrapping_mul(0xCB1A_B31F);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    h as f64 / u32::MAX as f64
}

/// Linear interpolation between `a` and `b` by `t`.
///
/// What: returns `a` at `t=0` and `b` at `t=1`.
/// How/why: `a + (b − a)·t`, the standard blend.
/// Units: dimensionless.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Value noise must stay in 0..1 and be repeatable.
    #[test]
    fn value_noise_is_bounded_and_deterministic() {
        for &(x, y, z) in &[(0.3, 1.7, 2.2), (10.5, -4.1, 0.0), (-3.3, 8.8, 5.5)] {
            let p = DVec3::new(x, y, z);
            let a = value_noise(p);
            let b = value_noise(p);
            assert!((0.0..=1.0).contains(&a), "noise out of range: {a}");
            assert_eq!(a, b, "noise must be deterministic");
        }
    }

    /// fBm must also stay in 0..1.
    #[test]
    fn fbm_is_bounded() {
        for i in 0..50 {
            let t = i as f64;
            let n = fbm(DVec3::new(t * 0.7, t * 1.3, t * 0.2), 6);
            assert!((0.0..=1.0).contains(&n), "fbm out of range: {n}");
        }
    }
}
