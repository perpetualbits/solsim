//! A procedural cloud map, made cheaply with fractal noise.
//!
//! Real clouds look the same at many scales — big puffs with smaller puffs on
//! them, down to wisps. We reproduce that with **fractional Brownian motion
//! (fBm)**: add up several "octaves" of smooth value noise, each one at twice the
//! frequency and half the strength of the one before. A handful of octaves already
//! gives a rich, cloud-like field, so it is very cheap — and we only bake it once,
//! into a texture, at start-up.
//!
//! We sample the noise on the **unit sphere** (using each pixel's 3-D direction)
//! rather than on the flat 2:1 map, so the band wraps seamlessly around the planet
//! and does not pinch at the poles. The fBm value is then turned into a coverage
//! (cloud/clear) with a soft threshold, and stored as the texture's alpha so the
//! renderer can drape it over the planet as a translucent shell.

use glam::DVec3;

use crate::noise::fbm;

use super::textures::{TEX_H, TEX_W};

/// How many octaves of noise to sum for the clouds.
///
/// What: the number of detail levels in the fractal.
/// How/why: each octave adds finer wisps at half the strength; 5–6 is plenty, and
/// more barely shows. Cost is one-time, so this is purely a look choice.
/// Units: a count.
const OCTAVES: u32 = 6;

/// Base feature scale of the clouds on the sphere.
///
/// What: how many big cloud blobs wrap around the planet.
/// How/why: the unit direction is multiplied by this before the first octave;
/// larger = smaller, more numerous blobs.
/// Units: dimensionless (cycles per radian-ish).
const BASE_FREQUENCY: f64 = 3.0;

/// Soft coverage thresholds applied to the (0..1) fBm value.
///
/// What: below `COVER_LO` the sky is clear, above `COVER_HI` it is solid cloud,
/// with a smooth edge between.
/// How/why: sliding these together/apart makes the planet more or less overcast;
/// this pair gives broken, partial cover like Earth from orbit.
/// Units: dimensionless (fBm value).
const COVER_LO: f64 = 0.46;
const COVER_HI: f64 = 0.62;

/// Strength of the domain warp that swirls the clouds.
///
/// What: how far the sample point is nudged by a second noise field before the
/// main fBm is read.
/// How/why: warping the domain turns blobby spots into wind-sheared streaks, which
/// reads as weather rather than cottage cheese.
/// Units: radians of displacement on the sphere (small).
const WARP_STRENGTH: f64 = 0.35;

/// Bake the procedural cloud coverage map (RGBA, `TEX_W`×`TEX_H`).
///
/// What: returns one texture-array layer where every pixel is white and its
/// **alpha** is the cloud coverage at that point on the planet.
/// How/why: for each pixel we find its direction on the unit sphere, swirl it with
/// a domain warp, read the fBm field there, and turn that into a soft 0..1
/// coverage; white-with-alpha lets the renderer blend it over the surface.
/// Principle: fBm (summed octaves of noise) is self-similar, like real clouds.
/// Units: returns `TEX_W·TEX_H·4` bytes of RGBA8.
pub fn bake_cloud_layer() -> Vec<u8> {
    let mut buf = vec![0u8; (TEX_W * TEX_H * 4) as usize];
    // Fixed offsets so the three warp fields and the main field do not line up.
    let o_main = DVec3::new(11.5, 3.2, 7.9);
    let o_wx = DVec3::new(1.7, 9.2, 4.4);
    let o_wy = DVec3::new(5.1, 2.8, 8.3);
    let o_wz = DVec3::new(6.6, 7.4, 1.9);

    for j in 0..TEX_H {
        // Pixel row → latitude (v=0 is the north pole, v=1 the south).
        let v = (j as f64 + 0.5) / TEX_H as f64;
        let lat = (0.5 - v) * std::f64::consts::PI;
        let (slat, clat) = lat.sin_cos();
        for i in 0..TEX_W {
            let u = (i as f64 + 0.5) / TEX_W as f64;
            let lon = u * std::f64::consts::TAU;
            let (slon, clon) = lon.sin_cos();
            // Direction of this pixel on the unit sphere.
            let dir = DVec3::new(clat * clon, clat * slon, slat);

            // Domain warp: nudge the sample point by a low-octave noise vector.
            let p = dir * BASE_FREQUENCY;
            let warp = DVec3::new(fbm(p + o_wx, 2), fbm(p + o_wy, 2), fbm(p + o_wz, 2))
                - DVec3::splat(0.5);
            let n = fbm(p + warp * WARP_STRENGTH + o_main, OCTAVES);

            let coverage = smoothstep(COVER_LO, COVER_HI, n);
            let a = (coverage * 255.0).round().clamp(0.0, 255.0) as u8;

            let idx = ((j * TEX_W + i) * 4) as usize;
            buf[idx] = 255;
            buf[idx + 1] = 255;
            buf[idx + 2] = 255;
            buf[idx + 3] = a;
        }
    }
    buf
}

/// A smooth 0→1 ramp between two edges (the GLSL `smoothstep`).
///
/// What: 0 below `e0`, 1 above `e1`, an S-curve in between.
/// How/why: clamp `(x−e0)/(e1−e0)` to 0..1, then shape it with `3t²−2t³` so the
/// transition has no hard corners.
/// Units: dimensionless.
fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The baked layer is the right size and genuinely varied (clear *and* cloudy).
    #[test]
    fn baked_layer_has_clouds_and_gaps() {
        let layer = bake_cloud_layer();
        assert_eq!(layer.len(), (TEX_W * TEX_H * 4) as usize);

        let mut min_a = 255u8;
        let mut max_a = 0u8;
        for px in layer.chunks_exact(4) {
            assert_eq!([px[0], px[1], px[2]], [255, 255, 255], "clouds are white");
            min_a = min_a.min(px[3]);
            max_a = max_a.max(px[3]);
        }
        assert!(
            min_a < 40,
            "should have clear sky somewhere (min alpha {min_a})"
        );
        assert!(
            max_a > 215,
            "should have solid cloud somewhere (max alpha {max_a})"
        );
    }
}
