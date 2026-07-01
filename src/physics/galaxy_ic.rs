//! Initial conditions for model galaxies and colliding pairs (the galaxy mode).
//!
//! A model galaxy here is a heavy **central mass** (bulge/black hole/inner halo,
//! all rolled into one point) surrounded by a thin **disk** of lighter particles.
//! Each disk particle is launched on a circular orbit — the speed that exactly
//! balances the inward pull of everything inside its radius, `v = √(G·M(<r)/r)` —
//! so the disk starts in rough equilibrium and holds together until something
//! disturbs it. Fly two such disks past each other and gravity's *tides* draw out
//! the long bridges and tails seen in real galaxy collisions.
//!
//! Phase 3 of the galaxy mode: build the starting state. Units are the caller's
//! own (pick `G = 1` and everything else scales); nothing is drawn yet.
#![allow(dead_code)]

use glam::Vec3;

use crate::rng::Rng;

/// The recipe for one model galaxy.
///
/// What: how many disk particles, how heavy the centre and the disk are, and the
/// disk's size, thinness and spin direction.
/// How/why: the disk radii follow an exponential profile (real disks fade
/// outward); `spin = +1` or `−1` sets prograde/retrograde rotation, which strongly
/// changes how tidal tails form in an encounter.
/// Units: masses in the caller's mass unit; `scale_radius`/`thickness` in its
/// length unit; `spin` dimensionless (±1).
pub struct GalaxyParams {
    pub n_disk: usize,
    pub central_mass: f32,
    pub disk_mass: f32,
    pub scale_radius: f32,
    pub thickness: f32,
    pub spin: f32,
}

/// Build one galaxy centred at the origin and at rest.
///
/// What: returns `(positions, velocities, masses)` with the central mass first,
/// then the disk particles.
/// How/why: disk radii are drawn from an exponential disk (the radial mass profile
/// `dM/dr ∝ r·e^{−r/r_d}` is a Gamma(2) law, sampled as the sum of two
/// exponentials), placed at a random angle in a thin layer. Each particle's speed
/// comes from the mass enclosed within its radius (centre + all disk particles
/// nearer in), so the disk is in centrifugal balance and stays a disk.
/// Principle: circular-orbit balance `G·M(<r)/r² = v²/r` — Kepler/Newton applied
/// shell by shell.
/// Units: as [`GalaxyParams`]; `g` the gravitational constant in the caller's units.
pub fn make_galaxy(p: &GalaxyParams, g: f32, seed: u64) -> (Vec<Vec3>, Vec<Vec3>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let m_part = if p.n_disk > 0 {
        p.disk_mass / p.n_disk as f32
    } else {
        0.0
    };
    let r_min = 0.05 * p.scale_radius; // keep particles off the singular centre

    // Sample each disk particle's radius, angle and height.
    let mut radius = Vec::with_capacity(p.n_disk);
    let mut angle = Vec::with_capacity(p.n_disk);
    let mut height = Vec::with_capacity(p.n_disk);
    for _ in 0..p.n_disk {
        // Gamma(2, r_d) = −r_d·(ln u₁ + ln u₂): an exponential disk.
        let r = (-p.scale_radius * ((rng.unit() as f32).max(1e-12).ln() + (rng.unit() as f32).max(1e-12).ln()))
            .max(r_min);
        radius.push(r);
        angle.push((rng.unit() as f32) * std::f32::consts::TAU);
        height.push(p.thickness * (rng.gaussian() as f32));
    }

    // Enclosed disk mass at each particle: sort by radius, count those nearer in.
    let mut order: Vec<usize> = (0..p.n_disk).collect();
    order.sort_by(|&a, &b| radius[a].total_cmp(&radius[b]));
    let mut enclosed = vec![0.0; p.n_disk];
    for (rank, &i) in order.iter().enumerate() {
        enclosed[i] = p.central_mass + m_part * rank as f32;
    }

    let mut pos = vec![Vec3::ZERO];
    let mut vel = vec![Vec3::ZERO];
    let mut mass = vec![p.central_mass];
    for i in 0..p.n_disk {
        let (s, c) = angle[i].sin_cos();
        pos.push(Vec3::new(radius[i] * c, radius[i] * s, height[i]));
        // Circular speed from the enclosed mass; tangential, sense set by `spin`.
        let v_circ = (g * enclosed[i] / radius[i]).sqrt();
        vel.push(Vec3::new(-s, c, 0.0) * (v_circ * p.spin));
        mass.push(m_part);
    }
    (pos, vel, mass)
}

/// Build two galaxies on a collision course.
///
/// What: returns the combined `(positions, velocities, masses)` of both galaxies,
/// the first placed to the left moving right and the second to the right moving
/// left, offset by an impact parameter and with the second disk tilted.
/// How/why: each galaxy is made at the origin, then rotated (galaxy B by
/// `inclination_b` about the x-axis, so the disks are not coplanar), shifted to
/// `∓separation/2` in x and `±impact/2` in y, and given equal and opposite bulk
/// velocities `±approach/2`. Tilting and the impact parameter are what turn a dull
/// head-on splat into bridges and tails.
/// Principle: a two-body hyperbolic/parabolic encounter, with each galaxy's stars
/// carried along and stirred by the other's tide.
/// Units: as [`make_galaxy`]; `separation`/`impact` lengths, `approach` a speed,
/// `inclination_b` in radians.
#[allow(clippy::too_many_arguments)]
pub fn colliding_pair(
    a: &GalaxyParams,
    b: &GalaxyParams,
    g: f32,
    separation: f32,
    approach: f32,
    impact: f32,
    inclination_b: f32,
    seed: u64,
) -> (Vec<Vec3>, Vec<Vec3>, Vec<f32>) {
    let (mut pa, mut va, ma) = make_galaxy(a, g, seed);
    let (pb, vb, mb) = make_galaxy(b, g, seed ^ 0xABCD_1234);

    // Galaxy A: shift left/up, move right.
    let off_a = Vec3::new(-0.5 * separation, 0.5 * impact, 0.0);
    let bulk_a = Vec3::new(0.5 * approach, 0.0, 0.0);
    for p in pa.iter_mut() {
        *p += off_a;
    }
    for v in va.iter_mut() {
        *v += bulk_a;
    }

    // Galaxy B: tilt its disk, then shift right/down and move left.
    let (si, ci) = inclination_b.sin_cos();
    let tilt = |q: Vec3| Vec3::new(q.x, q.y * ci - q.z * si, q.y * si + q.z * ci);
    let off_b = Vec3::new(0.5 * separation, -0.5 * impact, 0.0);
    let bulk_b = Vec3::new(-0.5 * approach, 0.0, 0.0);

    let mut pos = pa;
    let mut vel = va;
    let mut mass = ma;
    for ((p, v), m) in pb.iter().zip(vb.iter()).zip(mb.iter()) {
        pos.push(tilt(*p) + off_b);
        vel.push(tilt(*v) + bulk_b);
        mass.push(*m);
    }
    (pos, vel, mass)
}

/// A Milky-Way-and-Andromeda-like colliding pair in **physical units**, with a "Sun"
/// tracer tagged in the first galaxy — the initial state for the research mode.
///
/// What: returns `(positions, velocities, masses, sun_index)` in galactic units
/// (kpc, kpc/Myr, 10¹⁰ M☉), where `sun_index` is a disk star of galaxy A riding at
/// ~8 kpc — a stand-in for the Sun whose neighbourhood we watch during the collision.
/// How/why: two exponential disks with heavy centres (a bulge-plus-inner-halo proxy)
/// are tuned so a star at 8 kpc circles at the Sun's real ~220 km/s, then set on a
/// grazing approach. To keep the demo watchable we start them ~120 kpc apart closing
/// at ~150 km/s (the real pair is ~780 kpc and ~4.5 Gyr from first passage — the
/// physics of the passage is the same, we just skip the long coast). The Sun is the
/// disk particle of galaxy A whose radius is nearest 8 kpc.
/// Principle: a two-galaxy encounter in real units, so tides and densities read out in
/// numbers you can compare to the literature.
/// Units: kpc, kpc/Myr, 10¹⁰ M☉.
pub fn physical_pair(n_disk: usize, g: f32, seed: u64) -> (Vec<Vec3>, Vec<Vec3>, Vec<f32>, usize) {
    // Milky-Way-like: enclosed mass ~9×10¹⁰ M☉ inside 8 kpc → v_c ≈ 226 km/s there.
    let mw = GalaxyParams {
        n_disk,
        central_mass: 5.0, // bulge + inner-halo proxy, 5×10¹⁰ M☉
        disk_mass: 5.0,
        scale_radius: 3.0, // kpc
        thickness: 0.3,    // kpc disk scale height
        spin: 1.0,
    };
    // Andromeda-like: a bit heavier and larger.
    let m31 = GalaxyParams {
        n_disk,
        central_mass: 7.0,
        disk_mass: 7.0,
        scale_radius: 4.0,
        thickness: 0.3,
        spin: 1.0,
    };
    // separation 120 kpc, approach 0.15 kpc/Myr (~147 km/s), impact 30 kpc, tilt ~0.5 rad.
    let (pos, vel, mass) = colliding_pair(&mw, &m31, g, 120.0, 0.15, 30.0, 0.5, seed);

    // Tag the Sun: the galaxy-A disk star closest to 8 kpc from A's centre (particle 0).
    let centre = pos[0];
    let mut sun_index = 1;
    let mut best = f32::INFINITY;
    for (k, p) in pos[1..=n_disk].iter().enumerate() {
        let d = ((*p - centre).length() - 8.0).abs();
        if d < best {
            best = d;
            sun_index = k + 1;
        }
    }
    (pos, vel, mass, sun_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::galactic::{circular_speed, G_GAL, KMS_PER_KPC_MYR};
    use crate::physics::particles::Particles;

    /// The physical pair must tag a Sun near 8 kpc, orbiting at roughly 220 km/s.
    #[test]
    fn physical_pair_places_a_sun_like_star() {
        let g = G_GAL as f32;
        let (pos, vel, _mass, sun) = physical_pair(5000, g, 2024);
        // The Sun sits ~8 kpc from galaxy A's centre.
        let r = (pos[sun] - pos[0]).length();
        assert!((6.0..10.0).contains(&r), "Sun at {r} kpc, expected ~8");
        // Its speed relative to galaxy A's centre is ~200–260 km/s.
        let speed = ((vel[sun] - vel[0]).length() as f64) * KMS_PER_KPC_MYR;
        assert!((160.0..300.0).contains(&speed), "Sun speed {speed} km/s off");
        // Sanity: the circular-speed formula agrees at 8 kpc for ~9×10¹⁰ M☉ enclosed.
        let v_ref = circular_speed(9.0, 8.0) * KMS_PER_KPC_MYR;
        assert!((180.0..270.0).contains(&v_ref), "reference v_c {v_ref} km/s off");
    }

    /// The circular-velocity recipe: a massless disk around a point mass must keep
    /// a constant orbital radius (a Keplerian circle).
    #[test]
    fn test_particle_orbits_at_constant_radius() {
        let p = GalaxyParams {
            n_disk: 1,
            central_mass: 1.0,
            disk_mass: 0.0, // massless tracer → pure Kepler
            scale_radius: 1.0,
            thickness: 0.0,
            spin: 1.0,
        };
        let g = 1.0;
        let (pos, vel, mass) = make_galaxy(&p, g, 42);
        let r0 = pos[1].length();
        let mut sys = Particles::new(pos, vel, mass, 0.0, 1e-4, g);
        let period = std::f32::consts::TAU * (r0 * r0 * r0 / (g * 1.0)).sqrt();
        let steps = 2000;
        let dt = period / steps as f32;
        let mut min_r = f32::INFINITY;
        let mut max_r = 0.0_f32;
        for _ in 0..steps {
            sys.step(dt);
            let r = sys.pos[1].length();
            min_r = min_r.min(r);
            max_r = max_r.max(r);
        }
        assert!(
            (min_r - r0).abs() < 0.02 * r0 && (max_r - r0).abs() < 0.02 * r0,
            "radius wandered from {r0} to [{min_r}, {max_r}]"
        );
    }

    /// An isolated galaxy must stay roughly the same size — neither collapse to a
    /// point nor fly apart — over several dynamical times.
    #[test]
    fn isolated_galaxy_stays_bound() {
        let p = GalaxyParams {
            n_disk: 400,
            central_mass: 5.0,
            disk_mass: 1.0,
            scale_radius: 1.0,
            thickness: 0.05,
            spin: 1.0,
        };
        let g = 1.0;
        let (pos, vel, mass) = make_galaxy(&p, g, 7);

        let rms = |sys: &Particles| {
            // Cylindrical radius of the disk particles about the central mass.
            let c = sys.pos[0];
            let sum: f32 = sys.pos[1..]
                .iter()
                .map(|q| (*q - c).truncate().length_squared())
                .sum();
            (sum / (sys.pos.len() - 1) as f32).sqrt()
        };

        let mut sys = Particles::new(pos, vel, mass, 0.5, 0.05, g);
        let r_start = rms(&sys);
        // A few dynamical times at the disk scale.
        for _ in 0..800 {
            sys.step(0.01);
        }
        let ratio = rms(&sys) / r_start;
        assert!(
            (0.5..2.0).contains(&ratio),
            "disk size changed by ×{ratio} (start {r_start})"
        );
    }

    /// A colliding pair has the right particle count and its two centres sit at the
    /// intended offsets.
    #[test]
    fn colliding_pair_is_assembled() {
        let mk = |spin: f32| GalaxyParams {
            n_disk: 100,
            central_mass: 3.0,
            disk_mass: 1.0,
            scale_radius: 1.0,
            thickness: 0.05,
            spin,
        };
        let (pos, vel, mass) =
            colliding_pair(&mk(1.0), &mk(-1.0), 1.0, 8.0, 1.0, 1.5, 0.6, 99);
        assert_eq!(pos.len(), 2 * 101);
        assert_eq!(vel.len(), pos.len());
        assert_eq!(mass.len(), pos.len());
        // Central masses are the first particle of each galaxy.
        assert!((pos[0] - Vec3::new(-4.0, 0.75, 0.0)).length() < 1e-9);
        assert!((pos[101] - Vec3::new(4.0, -0.75, 0.0)).length() < 1e-9);
    }

    /// A colliding pair, stepped through the Barnes–Hut leapfrog, must stay finite
    /// (the full galaxy-mode dynamics path, minus the graphics).
    #[test]
    fn colliding_pair_stays_finite() {
        let mk = |spin: f32| GalaxyParams {
            n_disk: 200,
            central_mass: 4.0,
            disk_mass: 1.0,
            scale_radius: 1.0,
            thickness: 0.05,
            spin,
        };
        let (pos, vel, mass) =
            colliding_pair(&mk(1.0), &mk(1.0), 1.0, 12.0, 0.5, 2.5, 1.0, 1);
        let mut sys = Particles::new(pos, vel, mass, 0.6, 0.05, 1.0);
        for _ in 0..120 {
            sys.step(0.05);
        }
        for p in &sys.pos {
            assert!(p.is_finite(), "a particle went non-finite during the collision");
        }
        // The two galaxies should have moved toward each other (they were falling in).
        let sep = (sys.pos[0] - sys.pos[201]).length();
        assert!(sep < 12.0, "galaxies did not approach (separation {sep})");
    }
}
