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

use glam::Vec3;

use crate::physics::galaxy_ic::{colliding_pair, GalaxyParams};
use crate::physics::gpu::GpuNBody;
use crate::render::points::PointInstance;

/// Disk particles per galaxy. Raise toward 50 000 for the 100k-body target (it
/// runs slower per step); 30 000 keeps the collision smooth to watch.
const N_DISK: usize = 30_000;
/// Simulation time advanced per step (leapfrog is stable at this size).
const DT: f32 = 0.05;
/// Most simulation steps we will run in a single frame (the fast-forward cap).
const MAX_STEPS_PER_FRAME: u32 = 32;
/// Barnes–Hut opening angle (bigger = faster, a little rougher).
const THETA: f32 = 0.6;
/// Gravitational softening (a length), so close particles do not blow up.
const SOFTENING: f32 = 0.05;
/// Gravitational constant in these scale-free units.
const G: f32 = 1.0;

/// The running galaxy-collision simulation and how to colour it.
///
/// What: the GPU-resident particle system, a CPU mirror of the current positions
/// (copied back once per frame to draw), the split point between the two galaxies
/// (so we can colour them apart), and the elapsed simulation time.
/// How/why: particles `0..n_a` belong to galaxy A (its centre is particle 0),
/// the rest to galaxy B (its centre is particle `n_a`). The physics runs entirely on
/// the GPU (see [`GpuNBody`]); only `positions` comes back each frame.
/// Units: scale-free (`G = 1`).
pub struct GalaxyMode {
    sim: GpuNBody,
    positions: Vec<Vec3>,
    n_a: usize,
    time: f64,
    steps_per_frame: u32,
}

impl GalaxyMode {
    /// Set up two galaxies on a grazing collision course, resident on the GPU.
    ///
    /// What: builds the colliding pair and uploads it to the GPU integrator.
    /// How/why: two equal disks are placed a good distance apart, given a modest
    /// closing speed (below escape, so they actually fall together), an impact
    /// offset and a tilt on the second disk — the recipe for prominent tidal tails.
    /// The state then lives in GPU buffers; we keep the initial positions as the
    /// first frame's CPU mirror.
    /// Units: scale-free.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let galaxy = |spin: f32| GalaxyParams {
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
        let sim = GpuNBody::new(device, queue, &pos, &vel, &mass, THETA, SOFTENING, G);
        GalaxyMode {
            sim,
            positions: pos,
            n_a,
            time: 0.0,
            steps_per_frame: 1,
        }
    }

    /// Advance the simulation by one frame's worth of steps, then mirror positions.
    ///
    /// What: runs `steps_per_frame` GPU leapfrog steps and copies the new positions
    /// back to the CPU for drawing.
    /// How/why: each step is a single GPU submission; we read the position buffer back
    /// just once, after the last step, so fast-forward costs no extra copies.
    /// Units: none.
    pub fn step(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        for _ in 0..self.steps_per_frame {
            self.sim.step(device, queue, DT);
            self.time += DT as f64;
        }
        self.positions = self.sim.positions(device, queue);
    }

    /// Steps run per frame (the fast-forward setting).
    pub fn steps_per_frame(&self) -> u32 {
        self.steps_per_frame
    }

    /// Fast-forward: run more steps per frame (doubling, up to the cap).
    pub fn faster(&mut self) {
        self.steps_per_frame = (self.steps_per_frame * 2).min(MAX_STEPS_PER_FRAME);
    }

    /// Slow down: run fewer steps per frame (halving, at least one).
    pub fn slower(&mut self) {
        self.steps_per_frame = (self.steps_per_frame / 2).max(1);
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
    pub fn center(&self) -> Vec3 {
        (self.positions[0] + self.positions[self.n_a]) * 0.5
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
    pub fn points(&self, center: Vec3) -> Vec<PointInstance> {
        let mut out = Vec::with_capacity(self.positions.len());
        for (i, p) in self.positions.iter().enumerate() {
            let rel = *p - center;
            let core = i == 0 || i == self.n_a;
            let (color, size) = if core {
                ([1.0, 1.0, 0.9, 1.0], 8.0)
            } else if i < self.n_a {
                ([0.55, 0.72, 1.0, 0.62], 1.8)
            } else {
                ([1.0, 0.7, 0.42, 0.62], 1.8)
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
