//! Galactic-scale physics for the "what happens to the Solar System when two
//! galaxies collide?" research mode (Stage 1: read out the Sun's local environment).
//!
//! The galaxy collision and the Solar System live on wildly different scales — a
//! galaxy is ~10⁸ AU across and a merger takes a billion years, while Earth orbits in
//! one year at 1 AU. You cannot integrate both in one simulation, and you don't need
//! to: the galaxy never pulls on the planets directly (the Sun's grip is ~10¹⁵×
//! stronger at 1 AU). What the galaxy *does* do is change the Sun's **neighbourhood** —
//! and that neighbourhood is what can strip or stir up the weakly-bound **Oort cloud**
//! (out at ~0.03–1 pc), raining comets inward. So this module measures, along the
//! Sun's path through the collision, the two things that matter:
//!
//! 1. the **tidal field** — how hard the galaxy stretches the Solar System across its
//!    width (what slowly drags distant comets' orbits around), and
//! 2. the **local stellar density and velocity spread** — which set how often a
//!    passing star comes close enough to shower comets inward.
//!
//! This is a *one-way* coupling: the galaxy shapes the Sun's environment, but the
//! Oort cloud's feather-weight never pushes back on the galaxy. That is what makes the
//! enormous scale gap a feature (we integrate each part on its own clock) instead of a
//! wall.
//!
//! Everything here is in **f64** and **galactic units: kpc, Myr, 10¹⁰ M☉** — real
//! physics, so the numbers come out in units you can compare to the literature. (The
//! galaxy N-body itself stays f32 on the GPU; this is the precise CPU-side analysis of
//! its output.)
//!
//! *Resolution caveat:* our galaxy is made of massive "blob" particles (~10⁶ M☉
//! each), not individual stars, so the tidal field is only as smooth as the sim's
//! softening length, and individual star flybys are not resolved — the encounter rate
//! is an order-of-magnitude estimate from the local density, the way real Oort-cloud
//! studies do it.
#![allow(dead_code)]

use glam::{DMat3, DVec3};

/// Gravitational constant `G` in galactic units: kpc³ · (10¹⁰ M☉)⁻¹ · Myr⁻².
///
/// What: the strength of gravity in the units this module uses.
/// How/why: the tabulated value is `G = 4.30091·10⁻⁶ kpc·(km/s)²/M☉`. Converting the
/// speed `km/s → kpc/Myr` (÷977.8) squared, and the mass `M☉ → 10¹⁰ M☉` (×10¹⁰), gives
/// `G = 4.30091e-6 · (1/977.8)² · 1e10 ≈ 4.4985·10⁻²`.
/// Units: kpc³ (10¹⁰M☉)⁻¹ Myr⁻².
pub const G_GAL: f64 = 4.498_45e-2;

/// One parsec in kpc (the Oort cloud lives at ~0.03–1 pc).
pub const PC: f64 = 1.0e-3;
/// One astronomical unit in kpc (`1 AU = 1.496·10¹¹ m`, `1 kpc = 3.086·10¹⁹ m`).
pub const AU: f64 = 4.848_137e-9;
/// Multiply a speed in kpc/Myr by this to get km/s (`= 3.086·10¹⁶ km / 3.156·10¹³ s`).
pub const KMS_PER_KPC_MYR: f64 = 977.8;
/// A representative mean stellar mass, in 10¹⁰ M☉ (`≈ 0.5 M☉`), for turning a mass
/// density into a *number* density of stars.
pub const MEAN_STELLAR_MASS: f64 = 0.5e-10;

/// One body of the galaxy model: where it is, how fast, and how heavy.
///
/// Units: `pos` in kpc, `vel` in kpc/Myr, `mass` in 10¹⁰ M☉.
#[derive(Clone, Copy)]
pub struct Body {
    pub pos: DVec3,
    pub vel: DVec3,
    pub mass: f64,
}

/// Circular-orbit speed at radius `r` around an enclosed mass `m`.
///
/// What: the speed a body needs to circle at radius `r`.
/// How/why: set gravity equal to the centripetal pull, `G·m/r² = v²/r`, so
/// `v = √(G·m/r)`. It's how fast the Sun goes round the galaxy (~220 km/s at 8 kpc).
/// Principle: Newton's gravity providing exactly the circular acceleration.
/// Units: `m` in 10¹⁰ M☉, `r` in kpc; returns kpc/Myr.
pub fn circular_speed(m_enclosed: f64, r: f64) -> f64 {
    (G_GAL * m_enclosed / r).sqrt()
}

/// The galactic **tidal tensor** at a point, summed over all the galaxy's mass.
///
/// What: the 3×3 matrix `T` such that a small offset `δ` from the point feels a
/// *differential* pull `δa = T·δ` — i.e. how the galaxy's gravity varies across the
/// tiny span of the Solar System.
/// How/why: from the potential `Φ = −Σ G mᵢ/d`, the tensor is `Tᵢⱼ = −∂²Φ/∂xᵢ∂xⱼ`,
/// which works out to `Tᵢⱼ = Σ G mᵢ (3 sᵢsⱼ/d⁵ − δᵢⱼ/d³)` with `s = point − rᵢ` and
/// `d = |s|`. A `softening` term keeps it finite near a body and, set to the sim's
/// smoothing length, makes `T` reflect the *smooth* galaxy rather than one nearby blob.
/// Principle: tides are the *gradient* of gravity — the same reason the Moon raises two
/// bulges on Earth. For one mass at distance `r` the eigenvalues are `+2Gm/r³` along
/// the line to the mass (stretch) and `−Gm/r³` across it (squeeze).
/// Units: `bodies` in kpc / 10¹⁰ M☉, `softening` in kpc; `T` in Myr⁻² (an acceleration
/// per unit length).
pub fn tidal_tensor(point: DVec3, bodies: &[Body], softening: f64) -> DMat3 {
    let s2 = softening * softening;
    let mut t = DMat3::ZERO;
    for b in bodies {
        let s = point - b.pos;
        let d2 = s.length_squared() + s2;
        let d = d2.sqrt();
        let inv_d3 = 1.0 / (d2 * d);
        let inv_d5 = inv_d3 / d2;
        let gm = G_GAL * b.mass;
        // 3·G·m/d⁵ · (s ⊗ s): column j is s·sⱼ.
        let c = 3.0 * gm * inv_d5;
        let outer = DMat3::from_cols(s * (c * s.x), s * (c * s.y), s * (c * s.z));
        // ... minus G·m/d³ on the diagonal.
        t += outer - DMat3::from_diagonal(DVec3::splat(gm * inv_d3));
    }
    t
}

/// A single number for "how strong is the tide", from the tidal tensor.
///
/// What: the overall magnitude of the tidal field.
/// How/why: the Frobenius norm `√(Σ Tᵢⱼ²)` — a rotation-independent size of the matrix,
/// so it doesn't depend on how we orient our axes. It rises when the galaxy's pull
/// varies more steeply across the Solar System (a stronger, more disruptive tide).
/// Units: Myr⁻². Multiply by a comet's distance to get the tidal acceleration it feels.
pub fn tidal_strength(t: DMat3) -> f64 {
    let c = [t.x_axis, t.y_axis, t.z_axis];
    c.iter().map(|v| v.length_squared()).sum::<f64>().sqrt()
}

/// What the Sun's neighbourhood looks like right now: tide, crowding, and comet risk.
///
/// Units: see [`LocalEnv`].
#[derive(Clone, Copy, Debug)]
pub struct LocalEnv {
    /// Tidal-field strength (Frobenius norm of the tidal tensor), in Myr⁻².
    pub tidal_strength: f64,
    /// Local mass density, in 10¹⁰ M☉ / kpc³.
    pub density: f64,
    /// RMS speed of nearby bodies relative to the Sun, in km/s (typical flyby speed).
    pub dispersion_kms: f64,
    /// Order-of-magnitude rate of stellar passages within `close_au`, per Myr.
    pub encounter_rate: f64,
}

/// Measure the Sun's local environment from a galaxy snapshot.
///
/// What: computes the tidal strength at the Sun, the local density and velocity spread
/// from bodies within `sample_radius`, and an estimated rate of close stellar passages.
/// How/why: the tide comes from [`tidal_tensor`]. Density is the neighbour mass inside
/// a ball, `ρ = Σm / (4/3·π·R³)`. The dispersion is the RMS of neighbours' speeds
/// relative to the Sun — the speed at which stars sweep past. The encounter rate uses
/// the kinetic-theory estimate `rate ≈ n⋆ · v · π·b²`: turn the mass density into a
/// star number density `n⋆ = ρ / m⋆`, multiply by the flyby speed `v` and the target
/// area `π·b²` for passages within impact parameter `b = close_au`. That is exactly how
/// Oort-cloud studies estimate comet-shower frequency — our blobs can't resolve single
/// stars, but they give the density and speeds that set the rate.
/// Principle: a bigger, faster, denser crowd of stars means more near-misses, and each
/// near-miss can jostle the loosely-held outer Oort cloud.
/// Units: positions kpc, velocities kpc/Myr, masses 10¹⁰ M☉, `sample_radius` kpc,
/// `close_au` in AU, `softening` kpc. See [`LocalEnv`] for outputs.
pub fn local_environment(
    sun: Body,
    bodies: &[Body],
    sample_radius: f64,
    close_au: f64,
    softening: f64,
) -> LocalEnv {
    let tidal_strength = tidal_strength(tidal_tensor(sun.pos, bodies, softening));

    let r2 = sample_radius * sample_radius;
    let mut mass_in = 0.0;
    let mut sum_v2 = 0.0;
    let mut count = 0.0;
    for b in bodies {
        // Skip the Sun's own particle (a body at zero distance is "us", not a neighbour).
        let dr2 = (b.pos - sun.pos).length_squared();
        if dr2 > r2 || dr2 == 0.0 {
            continue;
        }
        mass_in += b.mass;
        sum_v2 += (b.vel - sun.vel).length_squared();
        count += 1.0;
    }
    let volume = (4.0 / 3.0) * std::f64::consts::PI * sample_radius * sample_radius * sample_radius;
    let density = mass_in / volume;
    let dispersion = if count > 0.0 { (sum_v2 / count).sqrt() } else { 0.0 };

    // Close-passage rate: n⋆ · v · π·b².
    let n_star = density / MEAN_STELLAR_MASS; // stars per kpc³
    let b_kpc = close_au * AU;
    let encounter_rate = n_star * dispersion * std::f64::consts::PI * b_kpc * b_kpc;

    LocalEnv {
        tidal_strength,
        density,
        dispersion_kms: dispersion * KMS_PER_KPC_MYR,
        encounter_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `G_GAL` must round-trip back to the tabulated `kpc·(km/s)²/M☉` value.
    #[test]
    fn g_matches_tabulated_value() {
        // Undo the unit conversion: kpc/Myr → km/s (×977.8²), 10¹⁰ M☉ → M☉ (÷1e10).
        let g_kpc_kms2_msun = G_GAL * KMS_PER_KPC_MYR * KMS_PER_KPC_MYR / 1.0e10;
        assert!(
            (g_kpc_kms2_msun - 4.30091e-6).abs() < 1e-8,
            "G converts to {g_kpc_kms2_msun}, expected 4.30091e-6"
        );
    }

    /// The Sun's ~220 km/s galactic orbit implies ~10¹¹ M☉ enclosed at 8 kpc.
    #[test]
    fn circular_speed_is_right_for_the_sun() {
        // Enclosed mass 10 (=10¹¹ M☉) at 8 kpc.
        let v = circular_speed(10.0, 8.0) * KMS_PER_KPC_MYR;
        assert!((150.0..280.0).contains(&v), "circular speed {v} km/s off the mark");
    }

    /// A single point mass gives the textbook tidal eigenvalues +2Gm/r³, −Gm/r³, −Gm/r³.
    #[test]
    fn tidal_tensor_of_point_mass() {
        let m = 3.0;
        let r = 5.0;
        let bodies = [Body { pos: DVec3::ZERO, vel: DVec3::ZERO, mass: m }];
        // Point on the x-axis, so x is the "radial" (stretch) direction.
        let t = tidal_tensor(DVec3::new(r, 0.0, 0.0), &bodies, 1e-9);
        let base = G_GAL * m / (r * r * r);
        assert!((t.x_axis.x - 2.0 * base).abs() < 1e-9 * base, "Txx wrong: {}", t.x_axis.x);
        assert!((t.y_axis.y + base).abs() < 1e-9 * base, "Tyy wrong: {}", t.y_axis.y);
        assert!((t.z_axis.z + base).abs() < 1e-9 * base, "Tzz wrong: {}", t.z_axis.z);
        // Off-diagonal terms vanish on axis.
        assert!(t.x_axis.y.abs() < 1e-9 * base, "off-diagonal not zero");
        // Frobenius norm = √(2²+1²+1²)·base = √6·base.
        assert!((tidal_strength(t) - 6.0_f64.sqrt() * base).abs() < 1e-9 * base);
    }

    /// A closer or heavier mass tides harder (∝ m/r³).
    #[test]
    fn tide_grows_closer_and_heavier() {
        let far = tidal_strength(tidal_tensor(
            DVec3::new(10.0, 0.0, 0.0),
            &[Body { pos: DVec3::ZERO, vel: DVec3::ZERO, mass: 1.0 }],
            1e-6,
        ));
        let near = tidal_strength(tidal_tensor(
            DVec3::new(5.0, 0.0, 0.0),
            &[Body { pos: DVec3::ZERO, vel: DVec3::ZERO, mass: 1.0 }],
            1e-6,
        ));
        // Halving the distance should raise the tide by 2³ = 8×.
        assert!((near / far - 8.0).abs() < 1e-3, "tide scaling wrong: {}", near / far);
    }

    /// Density, dispersion and encounter rate come out physically sensible for the Sun.
    #[test]
    fn local_environment_is_physical() {
        // A crude "solar neighbourhood": a uniform slab of blobs around the Sun with a
        // local density near the real ~0.1 M☉/pc³ and a ~40 km/s spread.
        let mut rng = crate::rng::Rng::new(0x50_1A5_u64);
        let sun = Body { pos: DVec3::new(8.0, 0.0, 0.0), vel: DVec3::ZERO, mass: 1e-4 };
        // Target ρ ≈ 0.1 M☉/pc³ = 1e-2 (10¹⁰M☉/kpc³). Within R = 0.5 kpc,
        // volume ≈ 0.52 kpc³, so total neighbour mass ≈ 5.2e-3; split over 2000 blobs.
        let n = 2000;
        let blob = 5.2e-3 / n as f64;
        let bodies: Vec<Body> = (0..n)
            .map(|_| {
                let dir = DVec3::new(
                    rng.unit() * 2.0 - 1.0,
                    rng.unit() * 2.0 - 1.0,
                    rng.unit() * 2.0 - 1.0,
                );
                let v = DVec3::new(
                    rng.gaussian() * 0.0409,
                    rng.gaussian() * 0.0409,
                    rng.gaussian() * 0.0409,
                ); // ~40 km/s dispersion in kpc/Myr
                Body { pos: sun.pos + dir.normalize_or_zero() * 0.4 * rng.unit(), vel: v, mass: blob }
            })
            .collect();

        let env = local_environment(sun, &bodies, 0.5, 1.0e5, 0.05);
        // Density within a factor of a few of the solar-neighbourhood value.
        assert!(
            (1e-3..1e-1).contains(&env.density),
            "density {} off (expected ~1e-2)",
            env.density
        );
        assert!((10.0..120.0).contains(&env.dispersion_kms), "dispersion {} km/s off", env.dispersion_kms);
        // A star within 10⁵ AU every ~fraction of a Myr to few Myr — order of magnitude.
        assert!(env.encounter_rate > 0.0 && env.encounter_rate < 1e3, "rate {} implausible", env.encounter_rate);
    }
}
