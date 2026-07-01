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

use crate::physics::galaxy_ic::{colliding_pair, GalaxyParams};
use crate::physics::gpu::GpuNBody;

/// Disk particles per galaxy → ~200k bodies total, a smooth default on a mid-range
/// GPU with the full GPU pipeline (radix sort + cooperative walk). Push it higher for
/// spectacle: ~250k/disk (500k total) still runs ~20–25 fps, a million is ~10 fps —
/// widen θ with `]` to claw some back.
const N_DISK: usize = 100_000;
/// Simulation time advanced per step (leapfrog is stable at this size).
const DT: f32 = 0.05;
/// Most simulation steps we will run in a single frame (the fast-forward cap).
const MAX_STEPS_PER_FRAME: u32 = 32;
/// Barnes–Hut opening angle (bigger = faster, a little rougher). 0.8 is a good
/// visual/speed balance; adjustable live with the `[` / `]` keys.
const THETA: f32 = 0.8;
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
    n_a: usize,
    time: f64,
    steps_per_frame: u32,
    theta: f32,
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
            n_a,
            time: 0.0,
            steps_per_frame: 1,
            theta: THETA,
        }
    }

    /// The GPU position buffer the renderer draws the cloud from (no CPU copy).
    pub fn pos_buffer(&self) -> &wgpu::Buffer {
        self.sim.pos_buffer()
    }

    /// Index of galaxy B's first particle (galaxy A is `0..n_a`), for colouring.
    pub fn n_a(&self) -> u32 {
        self.n_a as u32
    }

    /// Current Barnes–Hut opening angle θ (for the on-screen readout).
    pub fn theta(&self) -> f32 {
        self.theta
    }

    /// Widen θ (faster, rougher). Capped so the tree still means something.
    pub fn coarser(&mut self, queue: &wgpu::Queue) {
        self.theta = (self.theta + 0.1).min(1.5);
        self.sim.set_theta(queue, self.theta);
    }

    /// Narrow θ (slower, more accurate).
    pub fn finer(&mut self, queue: &wgpu::Queue) {
        self.theta = (self.theta - 0.1).max(0.2);
        self.sim.set_theta(queue, self.theta);
    }

    /// Advance the simulation by one frame's worth of steps, entirely on the GPU.
    ///
    /// What: runs `steps_per_frame` GPU leapfrog steps.
    /// How/why: each step is a single GPU submission; positions stay resident and are
    /// drawn straight from the GPU buffer, so nothing is copied back to the CPU.
    /// Units: none.
    pub fn step(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        for _ in 0..self.steps_per_frame {
            self.sim.step(device, queue, DT);
            self.time += DT as f64;
        }
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

}
