//! Our galaxy's band, plus the nearest neighbour galaxies.
//!
//! We cannot place a galaxy at its true distance — the nearest star is already
//! ~270 000 AU away and the galaxy is billions of AU across — so, like the
//! catalogue stars, galaxies are painted on the sky as *directions* only. We
//! scatter many faint background stars whose density is concentrated along the
//! galactic plane (a Gaussian in galactic latitude `b`), so that on the dark sky
//! their overlapping glows add up (the starfield uses additive blending) into the
//! soft band you see edge-on from inside our own galaxy's disk.
//!
//! The neighbour galaxies (Andromeda, the Magellanic Clouds, …) are made the same
//! way: each is a small Gaussian cloud of faint stars in an oriented ellipse at the
//! galaxy's real sky position, so it reads as a fuzzy elongated smudge that grows as
//! you zoom toward it.

use glam::DVec3;

use crate::noise::fbm;
use crate::render::starfield::StarInstance;
use crate::rng::Rng;
use crate::stars::project::{galactic_to_ecliptic, radec_to_ecliptic};

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
/// `BAND_MIN_BRIGHT` is the baseline along the whole band; `BAND_MAX_BRIGHT` is the
/// brightness at the heart of the bulge.
/// Units: linear-RGB brightness (dimensionless).
const BAND_MIN_BRIGHT: f64 = 0.12;
const BAND_MAX_BRIGHT: f64 = 0.30;

/// How tightly the bright, warm bulge is concentrated on the Galactic Centre.
///
/// What: the power applied to the bulge falloff.
/// How/why: the bulge weight is `(cos b·cos l)` (1 at the centre, dropping away);
/// raising it to this power shrinks the glow into a localized swelling around
/// Sagittarius instead of a band-long brightening. Higher = tighter.
/// Units: dimensionless exponent.
const BULGE_TIGHTNESS: f64 = 2.5;

/// Feature scale of the dust lanes (how finely the dark rifts are cut).
///
/// What: the noise frequency for the dust field.
/// How/why: the galactic direction is multiplied by this before the dust noise;
/// higher gives more, finer lanes.
/// Units: dimensionless.
const DUST_FREQUENCY: f64 = 5.0;

/// A fixed offset so the dust noise does not line up with anything else.
const DUST_OFFSET: DVec3 = DVec3::new(20.5, 13.1, 7.7);

/// Soft thresholds turning the dust noise into a clear↔dusty mix.
///
/// What: below `DUST_LO` the sky is fully dusty (dark), above `DUST_HI` it is clear.
/// How/why: sliding these sets how much of the band is eaten by dark lanes.
/// Units: dimensionless (noise value).
const DUST_LO: f64 = 0.42;
const DUST_HI: f64 = 0.52;

/// How far from the galactic plane the dust reaches, in degrees (Gaussian σ).
///
/// What: the dust only darkens stars near the plane (`b ≈ 0`), where it really lies.
/// How/why: real dust forms a thin layer in the disk, so the dark lanes split the
/// *core* of the band (the "Great Rift") and leave its edges alone.
/// Units: degrees.
const DUST_PLANE_SIGMA: f64 = 4.0;

/// Strongest fraction of a star's light the dust can absorb.
///
/// What: how dark the deepest lanes get (1.0 would be fully black).
/// How/why: just short of black so the lanes read as dark dust, not holes.
/// Units: dimensionless fraction.
const DUST_STRENGTH: f64 = 0.85;

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
        let (sl, cl) = l.to_radians().sin_cos();
        let (sb, cb) = b.to_radians().sin_cos();
        // Direction in the galactic frame (x → Galactic Centre, z → North Pole).
        let gal = DVec3::new(cb * cl, cb * sl, sb);

        // Bulge: a bright, warm swelling around the Galactic Centre. `gal.x` is 1 at
        // the centre and falls off in every direction; the power tightens it.
        let bulge = gal.x.max(0.0).powf(BULGE_TIGHTNESS);

        // Dust lanes: dark rifts where interstellar dust absorbs the starlight. Read
        // a noise field in galactic space and darken its dim patches — but only near
        // the plane (b ≈ 0), where the dust lies — so dark lanes split the band's
        // core (the Milky Way's "Great Rift").
        let clear = smoothstep(DUST_LO, DUST_HI, fbm(gal * DUST_FREQUENCY + DUST_OFFSET, 4));
        let in_plane = (-(b * b) / (2.0 * DUST_PLANE_SIGMA * DUST_PLANE_SIGMA)).exp();
        let absorption = DUST_STRENGTH * in_plane * (1.0 - clear);

        // Brighter toward the bulge, then dimmed by any dust in front.
        let base = BAND_MIN_BRIGHT + (BAND_MAX_BRIGHT - BAND_MIN_BRIGHT) * bulge;
        let bright = base * (0.6 + 0.8 * rng.unit()) * (1.0 - absorption);

        // Cool-neutral in the arms, warm gold toward the bulge.
        let warm = bulge;
        let color = [
            (bright * (0.82 + 0.42 * warm)) as f32,
            (bright * (0.85 + 0.10 * warm)) as f32,
            (bright * (0.95 - 0.38 * warm)) as f32,
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

/// Drawn-dot size range (pixels) for a neighbour-galaxy star.
const NEIGHBOR_SIZE_MIN: f64 = 2.2;
const NEIGHBOR_SIZE_MAX: f64 = 3.2;

/// One neighbouring galaxy, as a fuzzy elliptical smudge on the sky.
///
/// What: where a galaxy sits and how big, stretched and bright to draw it.
/// How/why: we scatter `points` faint stars in a Gaussian ellipse `major_deg` across
/// (squashed by `axis_ratio`, turned by `pa_deg`) at the galaxy's real (RA, Dec), so
/// it reads as a soft elongated patch. The positions are angular, so the smudge
/// scales correctly when you zoom.
/// Units: `ra_deg`/`dec_deg`/`major_deg`/`pa_deg` in degrees; `axis_ratio` and
/// `color` dimensionless; `bright` a linear-RGB level; `points` a count.
struct Neighbor {
    /// Kept for readability of the table; not used at runtime.
    #[allow(dead_code)]
    name: &'static str,
    ra_deg: f64,
    dec_deg: f64,
    major_deg: f64,
    axis_ratio: f64,
    pa_deg: f64,
    bright: f64,
    color: [f64; 3],
    points: usize,
}

/// The nearest, naked-eye galaxies (positions/sizes from catalogues, J2000).
///
/// What: Andromeda, Triangulum and the two Magellanic Clouds.
/// How/why: these are the galaxies actually visible to the eye; each is tinted
/// roughly by its stars (old-and-warm for the spirals' bulges, blue for the
/// star-forming Clouds) and sized to its real angular extent.
/// Units: see [`Neighbor`].
const NEIGHBORS: &[Neighbor] = &[
    Neighbor {
        name: "Andromeda (M31)",
        ra_deg: 10.68,
        dec_deg: 41.27,
        major_deg: 3.0,
        axis_ratio: 0.32,
        pa_deg: 35.0,
        bright: 0.16,
        color: [1.0, 0.95, 0.85],
        points: 280,
    },
    Neighbor {
        name: "Triangulum (M33)",
        ra_deg: 23.46,
        dec_deg: 30.66,
        major_deg: 1.2,
        axis_ratio: 0.6,
        pa_deg: 23.0,
        bright: 0.10,
        color: [0.92, 0.96, 1.0],
        points: 120,
    },
    Neighbor {
        name: "Large Magellanic Cloud",
        ra_deg: 80.89,
        dec_deg: -69.76,
        major_deg: 9.0,
        axis_ratio: 0.85,
        pa_deg: 170.0,
        bright: 0.14,
        color: [0.85, 0.92, 1.0],
        points: 320,
    },
    Neighbor {
        name: "Small Magellanic Cloud",
        ra_deg: 13.16,
        dec_deg: -72.8,
        major_deg: 5.0,
        axis_ratio: 0.5,
        pa_deg: 45.0,
        bright: 0.12,
        color: [0.85, 0.92, 1.0],
        points: 200,
    },
];

/// Build the faint star clouds for the neighbour galaxies.
///
/// What: returns [`StarInstance`]s for every galaxy in [`NEIGHBORS`].
/// How/why: for each galaxy we take its sky direction, build two tangent vectors
/// there, and scatter Gaussian-distributed points in an ellipse (σ = half the major
/// axis, squashed by the axis ratio, rotated by the position angle). Points near the
/// centre are also brightened, so the dense, bright core fades to a soft halo. A
/// fixed seed makes every galaxy identical each run.
/// Principle: an unresolved galaxy is the blended light of billions of stars; a
/// Gaussian cloud of faint additive points mimics that soft elliptical glow.
/// Units: directions are unit vectors; sizes in pixels; colours linear RGB.
pub fn neighbor_galaxies() -> Vec<StarInstance> {
    let mut rng = Rng::new(0x001C_EA11_DEAD_BEEF);
    let mut out = Vec::new();
    for g in NEIGHBORS {
        let d0 = radec_to_ecliptic(g.ra_deg, g.dec_deg);
        // Two unit tangents on the sky at the galaxy's centre.
        let helper = if d0.z.abs() < 0.95 {
            DVec3::Z
        } else {
            DVec3::X
        };
        let e1 = (helper - d0 * helper.dot(d0)).normalize();
        let e2 = d0.cross(e1);

        let sigma_major = (g.major_deg * 0.5).to_radians();
        let sigma_minor = sigma_major * g.axis_ratio;
        let (sp, cp) = g.pa_deg.to_radians().sin_cos();

        for _ in 0..g.points {
            let u = rng.gaussian(); // along the major axis (in σ units)
            let v = rng.gaussian(); // along the minor axis
                                    // Place the point, rotating the ellipse by the position angle.
            let a1 = sigma_major * (u * cp) - sigma_minor * (v * sp);
            let a2 = sigma_major * (u * sp) + sigma_minor * (v * cp);
            let dir = (d0 + e1 * a1 + e2 * a2).normalize();

            // Brighten the core (small u,v), fade the halo.
            let falloff = (-0.5 * (u * u + v * v)).exp();
            let bright = g.bright * (0.5 + 0.7 * rng.unit()) * (0.4 + 0.6 * falloff);
            let color = [
                (bright * g.color[0]) as f32,
                (bright * g.color[1]) as f32,
                (bright * g.color[2]) as f32,
            ];
            let size =
                (NEIGHBOR_SIZE_MIN + (NEIGHBOR_SIZE_MAX - NEIGHBOR_SIZE_MIN) * rng.unit()) as f32;
            out.push(StarInstance {
                dir: [dir.x as f32, dir.y as f32, dir.z as f32],
                size,
                color,
                _pad: 0.0,
            });
        }
    }
    out
}

/// A smooth 0→1 ramp between two edges (the GLSL `smoothstep`).
///
/// What: 0 below `e0`, 1 above `e1`, an S-curve in between.
/// How/why: clamp `(x−e0)/(e1−e0)` to 0..1, then shape it with `3t²−2t³` for a
/// soft transition — here, between clear sky and full dust.
/// Units: dimensionless.
fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
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

    /// Each neighbour galaxy makes the right number of unit-length points, clustered
    /// around its catalogue position.
    #[test]
    fn neighbor_galaxies_are_placed_and_clustered() {
        let total: usize = NEIGHBORS.iter().map(|g| g.points).sum();
        let stars = neighbor_galaxies();
        assert_eq!(stars.len(), total);
        for s in &stars {
            let len = (s.dir[0] * s.dir[0] + s.dir[1] * s.dir[1] + s.dir[2] * s.dir[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-4, "len = {len}");
        }

        // The first block of points belongs to Andromeda; they must sit near it.
        let m31 = &NEIGHBORS[0];
        let center = radec_to_ecliptic(m31.ra_deg, m31.dec_deg);
        let mut max_sep = 0.0_f64;
        for s in &stars[..m31.points] {
            let d = DVec3::new(s.dir[0] as f64, s.dir[1] as f64, s.dir[2] as f64);
            let sep = d.dot(center).clamp(-1.0, 1.0).acos().to_degrees();
            max_sep = max_sep.max(sep);
        }
        assert!(
            max_sep < 15.0,
            "M31 points stray {max_sep}° from its centre"
        );
    }
}
