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

use glam::DVec3;

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
    pub central_mass: f64,
    pub disk_mass: f64,
    pub scale_radius: f64,
    pub thickness: f64,
    pub spin: f64,
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
pub fn make_galaxy(p: &GalaxyParams, g: f64, seed: u64) -> (Vec<DVec3>, Vec<DVec3>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let m_part = if p.n_disk > 0 {
        p.disk_mass / p.n_disk as f64
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
        let r = (-p.scale_radius * (rng.unit().max(1e-12).ln() + rng.unit().max(1e-12).ln()))
            .max(r_min);
        radius.push(r);
        angle.push(rng.unit() * std::f64::consts::TAU);
        height.push(p.thickness * rng.gaussian());
    }

    // Enclosed disk mass at each particle: sort by radius, count those nearer in.
    let mut order: Vec<usize> = (0..p.n_disk).collect();
    order.sort_by(|&a, &b| radius[a].total_cmp(&radius[b]));
    let mut enclosed = vec![0.0; p.n_disk];
    for (rank, &i) in order.iter().enumerate() {
        enclosed[i] = p.central_mass + m_part * rank as f64;
    }

    let mut pos = vec![DVec3::ZERO];
    let mut vel = vec![DVec3::ZERO];
    let mut mass = vec![p.central_mass];
    for i in 0..p.n_disk {
        let (s, c) = angle[i].sin_cos();
        pos.push(DVec3::new(radius[i] * c, radius[i] * s, height[i]));
        // Circular speed from the enclosed mass; tangential, sense set by `spin`.
        let v_circ = (g * enclosed[i] / radius[i]).sqrt();
        vel.push(DVec3::new(-s, c, 0.0) * (v_circ * p.spin));
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
    g: f64,
    separation: f64,
    approach: f64,
    impact: f64,
    inclination_b: f64,
    seed: u64,
) -> (Vec<DVec3>, Vec<DVec3>, Vec<f64>) {
    let (mut pa, mut va, ma) = make_galaxy(a, g, seed);
    let (pb, vb, mb) = make_galaxy(b, g, seed ^ 0xABCD_1234);

    // Galaxy A: shift left/up, move right.
    let off_a = DVec3::new(-0.5 * separation, 0.5 * impact, 0.0);
    let bulk_a = DVec3::new(0.5 * approach, 0.0, 0.0);
    for p in pa.iter_mut() {
        *p += off_a;
    }
    for v in va.iter_mut() {
        *v += bulk_a;
    }

    // Galaxy B: tilt its disk, then shift right/down and move left.
    let (si, ci) = inclination_b.sin_cos();
    let tilt = |q: DVec3| DVec3::new(q.x, q.y * ci - q.z * si, q.y * si + q.z * ci);
    let off_b = DVec3::new(0.5 * separation, -0.5 * impact, 0.0);
    let bulk_b = DVec3::new(-0.5 * approach, 0.0, 0.0);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::particles::Particles;

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
        let period = std::f64::consts::TAU * (r0 * r0 * r0 / (g * 1.0)).sqrt();
        let steps = 2000;
        let dt = period / steps as f64;
        let mut min_r = f64::INFINITY;
        let mut max_r = 0.0_f64;
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
            let sum: f64 = sys.pos[1..]
                .iter()
                .map(|q| (*q - c).truncate().length_squared())
                .sum();
            (sum / (sys.pos.len() - 1) as f64).sqrt()
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
        let mk = |spin: f64| GalaxyParams {
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
        assert!((pos[0] - DVec3::new(-4.0, 0.75, 0.0)).length() < 1e-9);
        assert!((pos[101] - DVec3::new(4.0, -0.75, 0.0)).length() < 1e-9);
    }
}
