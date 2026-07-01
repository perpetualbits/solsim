//! The interactive "colliding galaxies" mode.
//!
//! A self-contained N-body world, separate from the solar system: two model
//! galaxies (a heavy centre plus a disk of particles) built by
//! [`crate::physics::galaxy_ic`], flown past each other and stepped with the
//! Barnes–Hut leapfrog engine. Each frame we advance the simulation and turn the
//! particles into a coloured point cloud for the renderer, so you watch gravity
//! draw out the tidal bridges and tails of a real galaxy collision.
//!
//! Everything runs in scale-free units (`G = 1`, disk scale length 1); the camera
//! just orbits the point cloud, so the absolute scale never matters.

use glam::DVec3;

use crate::physics::galaxy_ic::{colliding_pair, GalaxyParams};
use crate::physics::particles::Particles;
use crate::render::points::PointInstance;

/// Disk particles per galaxy. Raise toward 50 000 for the 100k-body target (it
/// runs slower per step); 30 000 keeps the collision smooth to watch.
const N_DISK: usize = 30_000;
/// Simulation time advanced per step (leapfrog is stable at this size).
const DT: f64 = 0.05;
/// Simulation steps advanced per rendered frame.
const STEPS_PER_FRAME: u32 = 1;
/// Barnes–Hut opening angle (bigger = faster, a little rougher).
const THETA: f64 = 0.6;
/// Gravitational softening (a length), so close particles do not blow up.
const SOFTENING: f64 = 0.05;
/// Gravitational constant in these scale-free units.
const G: f64 = 1.0;

/// The running galaxy-collision simulation and how to colour it.
///
/// What: the particle system, the split point between the two galaxies (so we can
/// colour them apart), and the elapsed simulation time.
/// How/why: particles `0..n_a` belong to galaxy A (its centre is particle 0),
/// the rest to galaxy B (its centre is particle `n_a`).
/// Units: scale-free (`G = 1`).
pub struct GalaxyMode {
    sim: Particles,
    n_a: usize,
    time: f64,
}

impl Default for GalaxyMode {
    fn default() -> Self {
        Self::new()
    }
}

impl GalaxyMode {
    /// Set up two galaxies on a grazing collision course.
    ///
    /// What: builds the colliding pair and primes the integrator.
    /// How/why: two equal disks are placed a good distance apart, given a modest
    /// closing speed (below escape, so they actually fall together), an impact
    /// offset and a tilt on the second disk — the recipe for prominent tidal tails.
    /// Units: scale-free.
    pub fn new() -> Self {
        let galaxy = |spin: f64| GalaxyParams {
            n_disk: N_DISK,
            central_mass: 4.0,
            disk_mass: 1.0,
            scale_radius: 1.0,
            thickness: 0.05,
            spin,
        };
        // separation 12, closing speed 0.5, impact 2.5, second disk tilted ~57°.
        let (pos, vel, mass) =
            colliding_pair(&galaxy(1.0), &galaxy(1.0), G, 12.0, 0.5, 2.5, 1.0, 0x6A1A_C71C);
        let n_a = N_DISK + 1;
        GalaxyMode {
            sim: Particles::new(pos, vel, mass, THETA, SOFTENING, G),
            n_a,
            time: 0.0,
        }
    }

    /// Advance the simulation by one frame's worth of steps.
    pub fn step(&mut self) {
        for _ in 0..STEPS_PER_FRAME {
            self.sim.step(DT);
            self.time += DT;
        }
    }

    /// Number of particles.
    pub fn len(&self) -> usize {
        self.sim.len()
    }

    /// Elapsed simulation time (scale-free units).
    pub fn time(&self) -> f64 {
        self.time
    }

    /// The midpoint between the two galaxy centres — a good camera target.
    pub fn center(&self) -> DVec3 {
        (self.sim.pos[0] + self.sim.pos[self.n_a]) * 0.5
    }

    /// Turn the particles into a coloured point cloud, relative to `center`.
    ///
    /// What: one [`PointInstance`] per particle (bright cores, faint blue disk for
    /// galaxy A and faint warm disk for galaxy B).
    /// How/why: additive faint dots overlap into a glowing sheet; two colours make
    /// the mixing and the tidal tails easy to follow. Positions are shifted by the
    /// camera target so they sit around the origin for the renderer.
    /// Units: `center` scale-free; output positions in the renderer's frame, sizes
    /// in pixels, colours linear RGBA.
    pub fn points(&self, center: DVec3) -> Vec<PointInstance> {
        let mut out = Vec::with_capacity(self.sim.len());
        for (i, p) in self.sim.pos.iter().enumerate() {
            let rel = (*p - center).as_vec3();
            let core = i == 0 || i == self.n_a;
            let (color, size) = if core {
                ([1.0, 1.0, 0.85, 1.0], 7.0)
            } else if i < self.n_a {
                ([0.45, 0.65, 1.0, 0.4], 1.6)
            } else {
                ([1.0, 0.65, 0.35, 0.4], 1.6)
            };
            out.push(PointInstance {
                pos: [rel.x, rel.y, rel.z],
                size,
                color,
            });
        }
        out
    }
}
