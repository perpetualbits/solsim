//! A softened N-body particle system, stepped with leapfrog and the Barnes–Hut
//! octree (the engine behind the coming galaxy mode).
//!
//! The bodies here are *sample particles* standing in for the smooth mass of a
//! galaxy, not real stars, so gravity is **softened** (a small `ε`) to model that
//! smoothness and keep close encounters tame. Time is advanced with **leapfrog**
//! (kick-drift-kick): nudge the velocities half a step, drift the positions a full
//! step, recompute the forces, then nudge the velocities the other half. Leapfrog
//! is *symplectic* — it does not slowly gain or lose energy the way a naive method
//! does — which is exactly what a long galaxy simulation needs.
//!
//! Phase 2 of the galaxy mode: the integrator, checked against known physics
//! (a closed two-body orbit, conserved energy and momentum). Not drawn yet.
#![allow(dead_code)]

use glam::DVec3;

use super::octree::Octree;

/// A cloud of gravitating particles and the settings that move them.
///
/// What: positions, velocities and masses of every particle, the current
/// acceleration (kept in step with the positions), and the force settings.
/// How/why: leapfrog reuses the end-of-step acceleration as the start of the next,
/// so we store it rather than recompute it twice.
/// Units: caller's own (e.g. kpc, Myr, solar masses); `theta` dimensionless,
/// `softening` a length, `g` the gravitational constant in those units.
pub struct Particles {
    pub pos: Vec<DVec3>,
    pub vel: Vec<DVec3>,
    pub mass: Vec<f64>,
    acc: Vec<DVec3>,
    pub theta: f64,
    pub softening: f64,
    pub g: f64,
}

impl Particles {
    /// Create a system and compute its starting accelerations.
    ///
    /// What: bundles the state and primes `acc` so the first [`step`](Self::step)
    /// is a valid leapfrog kick.
    /// How/why: leapfrog needs the acceleration *before* the first half-kick.
    /// Units: as the struct.
    pub fn new(
        pos: Vec<DVec3>,
        vel: Vec<DVec3>,
        mass: Vec<f64>,
        theta: f64,
        softening: f64,
        g: f64,
    ) -> Self {
        let acc = Octree::accelerations(&pos, &mass, theta, softening, g);
        Particles {
            pos,
            vel,
            mass,
            acc,
            theta,
            softening,
            g,
        }
    }

    /// Number of particles.
    pub fn len(&self) -> usize {
        self.pos.len()
    }

    /// Whether the system is empty.
    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    /// Advance the whole system by one time step `dt` (leapfrog kick-drift-kick).
    ///
    /// What: moves every particle forward by `dt`, keeping `acc` in sync.
    /// How/why: `v ← v + a·½dt` (kick with the stored acceleration), `x ← x + v·dt`
    /// (drift), recompute `a` at the new positions with the octree, then
    /// `v ← v + a·½dt` (kick with the new acceleration). Splitting the kick around
    /// the drift is what makes leapfrog time-reversible and energy-stable.
    /// Principle: the same `r¨ = a` as any gravity integrator, but arranged so that
    /// errors cancel over a step instead of accumulating.
    /// Units: `dt` in the caller's time unit.
    pub fn step(&mut self, dt: f64) {
        let half = 0.5 * dt;
        for (v, a) in self.vel.iter_mut().zip(&self.acc) {
            *v += *a * half;
        }
        for (p, v) in self.pos.iter_mut().zip(&self.vel) {
            *p += *v * dt;
        }
        self.acc = Octree::accelerations(&self.pos, &self.mass, self.theta, self.softening, self.g);
        for (v, a) in self.vel.iter_mut().zip(&self.acc) {
            *v += *a * half;
        }
    }

    /// Total kinetic energy `Σ ½·mᵢ·vᵢ²`.
    ///
    /// Units: caller's mass·length²·time⁻².
    pub fn kinetic_energy(&self) -> f64 {
        self.mass
            .iter()
            .zip(&self.vel)
            .map(|(m, v)| 0.5 * m * v.length_squared())
            .sum()
    }

    /// Total (softened) gravitational potential energy.
    ///
    /// What: `−G·Σ_{i<j} mᵢ·mⱼ / √(r_ij² + ε²)`.
    /// How/why: the pairwise potential of the *softened* force, so it matches the
    /// dynamics; this is an O(N²) sum meant for checking small systems, not for the
    /// hot loop.
    /// Units: caller's mass·length²·time⁻².
    pub fn potential_energy(&self) -> f64 {
        let soft2 = self.softening * self.softening;
        let mut pe = 0.0;
        for i in 0..self.pos.len() {
            for j in (i + 1)..self.pos.len() {
                let r = (self.pos[j] - self.pos[i]).length_squared() + soft2;
                pe -= self.g * self.mass[i] * self.mass[j] / r.sqrt();
            }
        }
        pe
    }

    /// Total energy (kinetic + potential); should stay nearly constant.
    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy()
    }

    /// Total momentum `Σ mᵢ·vᵢ`; conserved exactly for exact (θ = 0) forces.
    pub fn total_momentum(&self) -> DVec3 {
        self.mass
            .iter()
            .zip(&self.vel)
            .fold(DVec3::ZERO, |s, (m, v)| s + *v * *m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny deterministic RNG (SplitMix64) returning values in [0, 1).
    fn rng(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// A two-body circular orbit must keep a constant separation and return home.
    ///
    /// Two equal masses set on a circular mutual orbit should, after one period,
    /// have kept their separation (here 1) fixed and come back to the start.
    #[test]
    fn two_body_circular_orbit_is_closed() {
        let g: f64 = 1.0;
        let (m1, m2): (f64, f64) = (1.0, 1.0);
        let d: f64 = 1.0;
        let mu = g * (m1 + m2);
        let v_rel = (mu / d).sqrt();
        // Place on the x-axis about the (stationary) centre of mass.
        let x1 = -d * m2 / (m1 + m2);
        let x2 = d * m1 / (m1 + m2);
        let v1 = -(m2 / (m1 + m2)) * v_rel;
        let v2 = (m1 / (m1 + m2)) * v_rel;
        let pos = vec![DVec3::new(x1, 0.0, 0.0), DVec3::new(x2, 0.0, 0.0)];
        let vel = vec![DVec3::new(0.0, v1, 0.0), DVec3::new(0.0, v2, 0.0)];
        let mut sys = Particles::new(pos, vel, vec![m1, m2], 0.0, 1e-3, g);

        let start = sys.pos[0];
        let period = std::f64::consts::TAU * (d * d * d / mu).sqrt();
        let steps = 4000;
        let dt = period / steps as f64;
        let mut min_sep = f64::INFINITY;
        let mut max_sep = 0.0_f64;
        for _ in 0..steps {
            sys.step(dt);
            let sep = (sys.pos[1] - sys.pos[0]).length();
            min_sep = min_sep.min(sep);
            max_sep = max_sep.max(sep);
        }
        assert!(
            (min_sep - 1.0).abs() < 0.02 && (max_sep - 1.0).abs() < 0.02,
            "separation drifted to [{min_sep}, {max_sep}]"
        );
        assert!(
            (sys.pos[0] - start).length() < 0.02,
            "body did not return after one period"
        );
    }

    /// Leapfrog must conserve total energy to a small bounded fraction.
    #[test]
    fn energy_is_conserved() {
        let mut rng = rng(0x00C0_FFEE_1234_5678);
        let n = 60;
        let mut pos = Vec::new();
        let mut vel = Vec::new();
        for _ in 0..n {
            pos.push(DVec3::new(rng() * 2.0 - 1.0, rng() * 2.0 - 1.0, rng() * 2.0 - 1.0));
            vel.push(DVec3::new(rng() - 0.5, rng() - 0.5, rng() - 0.5) * 0.1);
        }
        let mut sys = Particles::new(pos, vel, vec![1.0; n], 0.4, 0.1, 1.0);
        let e0 = sys.total_energy();
        for _ in 0..1000 {
            sys.step(0.001);
        }
        let drift = (sys.total_energy() - e0).abs() / e0.abs().max(1e-12);
        assert!(drift < 0.02, "energy drifted by {drift}");
    }

    /// With exact forces (θ = 0) total momentum is conserved to rounding.
    #[test]
    fn momentum_is_conserved() {
        let mut rng = rng(0x0000_ABCD_0000_0001);
        let n = 40;
        let mut pos = Vec::new();
        let mut vel = Vec::new();
        for _ in 0..n {
            pos.push(DVec3::new(rng() * 2.0 - 1.0, rng() * 2.0 - 1.0, rng() * 2.0 - 1.0));
            vel.push(DVec3::new(rng() - 0.5, rng() - 0.5, rng() - 0.5) * 0.2);
        }
        let mut sys = Particles::new(pos, vel, vec![1.0; n], 0.0, 0.1, 1.0);
        let p0 = sys.total_momentum();
        for _ in 0..300 {
            sys.step(0.002);
        }
        assert!(
            (sys.total_momentum() - p0).length() < 1e-9,
            "momentum changed by {}",
            (sys.total_momentum() - p0).length()
        );
    }
}
