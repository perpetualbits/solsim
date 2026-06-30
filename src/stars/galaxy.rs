//! The procedural Milky Way band.
//!
//! We cannot place a galaxy at its true distance — the nearest star is already
//! ~270 000 AU away and the galaxy is billions of AU across — so, like the
//! catalogue stars, the Milky Way is painted on the sky as *directions* only. We
//! scatter many faint background stars whose density is concentrated along the
//! galactic plane (a Gaussian in galactic latitude `b`), so that on the dark sky
//! their overlapping glows add up (the starfield uses additive blending) into the
//! soft band you see edge-on from inside our own galaxy's disk.

use crate::render::starfield::StarInstance;
use crate::stars::project::galactic_to_ecliptic;

/// How many faint stars make up the band.
///
/// What: the size of the procedural Milky Way point cloud.
/// How/why: more points give a smoother, denser glow (the band is the overlap of
/// many faint dots); tens of thousands cost nothing to draw. Bumping this thickens
/// and brightens the band evenly.
/// Units: a count.
const BAND_STAR_COUNT: usize = 30_000;

/// Vertical spread of the band, as the Gaussian σ of galactic latitude `b`.
///
/// What: how thick the band is on the sky.
/// How/why: most stars land within ≈ ±2σ of the galactic plane; ~6° matches the
/// naked-eye impression of a band a couple of tens of degrees wide.
/// Units: degrees.
const BAND_LATITUDE_SIGMA_DEG: f64 = 6.0;

/// Faintest and brightest per-star glow values (for additive blending).
///
/// What: the dim end (toward the anticentre) and bright end (toward the bulge) of
/// each band star's colour.
/// How/why: kept well below the catalogue stars (~1.0) so the band reads as a
/// diffuse glow, not as competing points; the bulge near the Galactic Centre is a
/// touch brighter than the outer arms. With additive blending many of these sum up
/// in the dense band.
/// Units: linear-RGB brightness (dimensionless).
const BAND_MIN_BRIGHT: f64 = 0.10;
const BAND_MAX_BRIGHT: f64 = 0.22;

/// Smallest and largest drawn dot size for a band star.
///
/// What: the pixel-size range of the band stars.
/// How/why: must be ≳ 1 px or the sprite falls between pixels and renders nothing;
/// kept just under the catalogue's smallest star (`MIN_SIZE` = 3.6 px) so the band
/// reads as fainter, softer background haze, with many overlapping into a glow.
/// Units: pixels.
const BAND_SIZE_MIN: f64 = 2.4;
const BAND_SIZE_MAX: f64 = 3.4;

/// Build the faint star cloud that forms the Milky Way band.
///
/// What: returns a list of [`StarInstance`]s concentrated along the galactic plane.
/// How/why: for each star we pick a uniform galactic longitude `l`, a Gaussian
/// latitude `b` (so the cloud hugs the plane), then dim it and tint it slightly
/// warmer toward the Galactic Centre (`l ≈ 0`, the bulge) and cooler in the arms;
/// finally [`galactic_to_ecliptic`] turns `(l, b)` into the sky direction the
/// renderer needs. A fixed seed makes the band identical every run.
/// Principle: the band is the combined light of countless unresolved stars in the
/// disk we sit inside; many faint additive glows near `b = 0` reproduce that look.
/// Units: returned directions are unit vectors; sizes in pixels; colours linear RGB.
pub fn milky_way_band() -> Vec<StarInstance> {
    let mut rng = Rng::new(0x5EED_1234_ABCD_0001);
    let mut out = Vec::with_capacity(BAND_STAR_COUNT);
    for _ in 0..BAND_STAR_COUNT {
        let l = rng.unit() * 360.0;
        // Gaussian latitude: the band is thin near the plane, thinning out above it.
        let b = BAND_LATITUDE_SIGMA_DEG * rng.gaussian();

        // Brighter toward the Galactic Centre (l ≈ 0): `center` runs 1 → 0.
        let center = 0.5 + 0.5 * l.to_radians().cos();
        let base = BAND_MIN_BRIGHT + (BAND_MAX_BRIGHT - BAND_MIN_BRIGHT) * center;
        let bright = base * (0.6 + 0.8 * rng.unit()); // per-star brightness jitter

        // Slightly warm (yellow) toward the bulge, slightly cool (blue) in the arms.
        let warm = center;
        let color = [
            (bright * (0.85 + 0.20 * warm)) as f32,
            (bright * 0.88) as f32,
            (bright * (0.95 - 0.10 * warm)) as f32,
        ];
        let size = (BAND_SIZE_MIN + (BAND_SIZE_MAX - BAND_SIZE_MIN) * rng.unit()) as f32;

        let dir = galactic_to_ecliptic(l, b);
        out.push(StarInstance {
            dir: [dir.x as f32, dir.y as f32, dir.z as f32],
            size,
            color,
            _pad: 0.0,
        });
    }
    out
}

/// A tiny deterministic random-number generator (SplitMix64).
///
/// What: produces a repeatable stream of pseudo-random numbers from a seed.
/// How/why: we need scattered-but-reproducible star positions without pulling in a
/// random-number crate; SplitMix64 is a few lines and good enough for placing dots.
/// Principle: hashing a steadily increasing counter yields well-mixed bits.
/// Units: none.
struct Rng(u64);

impl Rng {
    /// Start the generator from a fixed seed.
    ///
    /// What: creates an `Rng` whose stream is fully determined by `seed`.
    /// How/why: a constant seed means the band looks the same every run.
    /// Units: none.
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    /// Return the next 64 random bits (the SplitMix64 step).
    ///
    /// What: advances the internal counter and hashes it.
    /// How/why: add the golden-ratio constant, then two xor-shift-multiply rounds
    /// scramble the bits so consecutive outputs look independent.
    /// Units: none (raw bits).
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform random number in the half-open interval [0, 1).
    ///
    /// What: a fractional random value.
    /// How/why: take the top 53 bits (the mantissa width of `f64`) and divide by
    /// 2⁵³, giving an evenly spaced value in [0, 1).
    /// Units: dimensionless.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A standard-normal random number (mean 0, standard deviation 1).
    ///
    /// What: a bell-curve-distributed value, used for the band's latitude.
    /// How/why: the Box–Muller transform turns two uniforms `u₁, u₂` into a normal
    /// value `√(−2·ln u₁)·cos(2π·u₂)`.
    /// Principle: Box–Muller maps a uniform square onto a Gaussian via polar form.
    /// Units: dimensionless (multiply by σ for a chosen spread).
    fn gaussian(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12); // avoid ln(0)
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    /// The band must contain the requested number of stars and be reproducible.
    #[test]
    fn band_is_deterministic_and_complete() {
        let a = milky_way_band();
        let b = milky_way_band();
        assert_eq!(a.len(), BAND_STAR_COUNT);
        assert_eq!(a[0].dir, b[0].dir, "same seed must give the same band");
        assert_eq!(a[BAND_STAR_COUNT - 1].dir, b[BAND_STAR_COUNT - 1].dir);
    }

    /// Every drawn direction is (very nearly) a unit vector.
    #[test]
    fn band_directions_are_unit_length() {
        for s in milky_way_band() {
            let len = (s.dir[0] * s.dir[0] + s.dir[1] * s.dir[1] + s.dir[2] * s.dir[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-4, "len = {len}");
        }
    }

    /// The band really is concentrated near the galactic plane.
    ///
    /// We measure each star's galactic latitude back out as the angle from the
    /// galactic plane (its sine is the dot product with the pole direction) and
    /// check the average stays small — far thinner than an even sky would give
    /// (a uniform sphere averages 57.3° × … ≈ 32.7° of |latitude|).
    #[test]
    fn band_hugs_the_galactic_plane() {
        let pole = galactic_to_ecliptic(0.0, 90.0); // NGP direction, in ecliptic
        let band = milky_way_band();
        let mut sum_abs_b = 0.0;
        for s in &band {
            let dir = DVec3::new(s.dir[0] as f64, s.dir[1] as f64, s.dir[2] as f64);
            sum_abs_b += dir.dot(pole).clamp(-1.0, 1.0).asin().abs();
        }
        let mean_deg = (sum_abs_b / band.len() as f64).to_degrees();
        assert!(mean_deg < 12.0, "band too thick: mean |b| = {mean_deg}°");
    }
}
