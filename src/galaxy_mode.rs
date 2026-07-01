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

use crate::physics::galactic::{self, Body, LocalEnv, G_GAL};
use crate::physics::galaxy_ic::{colliding_pair, physical_pair, GalaxyParams};
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

// --- Research mode (physical units: kpc, Myr, 10¹⁰ M☉) ---
/// Disk particles per galaxy in research mode. Fewer than the visual toy, because the
/// per-frame environment probe reads the state back and sums over every particle.
const N_DISK_RESEARCH: usize = 80_000;
/// Timestep in Myr — small against the Sun's ~220 Myr galactic orbit, so leapfrog is
/// stable.
const DT_MYR: f32 = 1.0;
/// θ for the research run (a touch tighter than the visual toy for a cleaner tide).
const THETA_RESEARCH: f32 = 0.7;
/// Softening in kpc — also the scale below which the tidal field is smoothed.
const SOFTENING_KPC: f32 = 0.2;
/// Radius (kpc) of the ball we sample for the Sun's local density and velocity spread.
const SAMPLE_RADIUS_KPC: f64 = 1.0;
/// Impact parameter (AU) for the "close stellar passage" encounter-rate estimate
/// (~0.5 pc, the outer-Oort-cloud danger zone).
const CLOSE_AU: f64 = 1.0e5;

/// One measurement of the Sun's neighbourhood, tagged with the time it was taken.
#[derive(Clone, Copy)]
pub struct EnvSample {
    /// Simulation time, Myr.
    pub time_myr: f64,
    /// The measured local environment.
    pub env: LocalEnv,
}

/// The research-mode extras: the tagged Sun and the history of its environment.
///
/// What: which particle is the Sun, a CPU copy of the masses (needed by the probe),
/// and the recorded time series of the Sun's local environment.
/// Units: physical (kpc, Myr, 10¹⁰ M☉).
pub struct Research {
    sun_index: usize,
    masses: Vec<f32>,
    history: Vec<EnvSample>,
}

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
    dt: f32,
    research: Option<Research>,
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
            dt: DT,
            research: None,
        }
    }

    /// Set up the **research** run: a physical Milky-Way–Andromeda collision with a
    /// tagged Sun, ready to record the Sun's changing neighbourhood.
    ///
    /// What: builds the physical colliding pair (kpc/Myr/10¹⁰ M☉), uploads it to the
    /// GPU integrator, tags the Sun, and takes a first environment reading.
    /// How/why: same GPU engine as the visual mode, but in real units and with the
    /// per-frame probe switched on. The masses are copied to the CPU because the probe
    /// needs them (the GPU keeps only positions/velocities resident).
    /// Units: physical.
    pub fn new_research(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let g = G_GAL as f32;
        let (pos, vel, mass, sun_index) = physical_pair(N_DISK_RESEARCH, g, 0x501A_5EED_u64);
        let n_a = N_DISK_RESEARCH + 1;
        let sim = GpuNBody::new(device, queue, &pos, &vel, &mass, THETA_RESEARCH, SOFTENING_KPC, g);
        let mut mode = GalaxyMode {
            sim,
            n_a,
            time: 0.0,
            steps_per_frame: 1,
            theta: THETA_RESEARCH,
            dt: DT_MYR,
            research: Some(Research { sun_index, masses: mass, history: Vec::new() }),
        };
        // Record the starting (quiescent) environment as the baseline.
        mode.probe(device, queue);
        mode
    }

    /// Measure the Sun's local environment now and append it to the history.
    ///
    /// What: reads positions and velocities back from the GPU, builds `f64` bodies, and
    /// calls [`galactic::local_environment`] at the Sun.
    /// How/why: only in research mode, once per frame — a full read-back plus an O(N)
    /// sum, cheap enough at this particle count and off the visual-mode hot path.
    /// Units: physical.
    fn probe(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let Some(research) = self.research.as_mut() else {
            return;
        };
        let pos = self.sim.positions(device, queue);
        let vel = self.sim.velocities(device, queue);
        let bodies: Vec<Body> = pos
            .iter()
            .zip(&vel)
            .zip(&research.masses)
            .map(|((p, v), m)| Body {
                pos: p.as_dvec3(),
                vel: v.as_dvec3(),
                mass: *m as f64,
            })
            .collect();
        let sun = bodies[research.sun_index];
        let env = galactic::local_environment(
            sun,
            &bodies,
            SAMPLE_RADIUS_KPC,
            CLOSE_AU,
            SOFTENING_KPC as f64,
        );
        research.history.push(EnvSample { time_myr: self.time, env });
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
    /// What: runs `steps_per_frame` GPU leapfrog steps; in research mode also samples
    /// the Sun's environment once, after the steps.
    /// How/why: each step is a single GPU submission; positions stay resident and are
    /// drawn straight from the GPU buffer. The visual mode copies nothing back; the
    /// research mode reads back once per frame for the probe.
    /// Units: `dt` in the mode's time unit (scale-free, or Myr in research mode).
    pub fn step(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        for _ in 0..self.steps_per_frame {
            self.sim.step(device, queue, self.dt);
            self.time += self.dt as f64;
        }
        if self.research.is_some() {
            self.probe(device, queue);
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

    /// Elapsed simulation time (scale-free units, or Myr in research mode).
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Whether this is the physical research run (vs the visual toy).
    pub fn is_research(&self) -> bool {
        self.research.is_some()
    }

    /// The tagged Sun's particle index, if this is a research run (for highlighting).
    pub fn sun_index(&self) -> Option<u32> {
        self.research.as_ref().map(|r| r.sun_index as u32)
    }

    /// The recorded time series of the Sun's environment (research mode).
    pub fn history(&self) -> &[EnvSample] {
        self.research.as_ref().map(|r| r.history.as_slice()).unwrap_or(&[])
    }

    /// The most recent environment reading, and the baseline (first) reading — so the
    /// UI can show "how many times worse than a quiet galaxy is it now".
    pub fn latest_and_baseline(&self) -> Option<(LocalEnv, LocalEnv)> {
        let h = &self.research.as_ref()?.history;
        Some((h.last()?.env, h.first()?.env))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The research run must build, step on the GPU, and read a *physical* environment
    /// back at the Sun — the whole Stage-1 chain (physical IC → GPU sim → probe).
    #[test]
    fn research_probe_reads_a_physical_environment() {
        let Some((device, queue)) = crate::physics::gpu::headless_device() else {
            eprintln!("no GPU; skipping");
            return;
        };
        let mut gm = GalaxyMode::new_research(&device, &queue);
        assert!(gm.is_research());
        assert!(gm.sun_index().is_some_and(|s| (s as usize) < gm.len()));

        for _ in 0..3 {
            gm.step(&device, &queue);
        }
        let (latest, baseline) = gm.latest_and_baseline().expect("research history");
        assert!(gm.history().len() >= 2, "history should accumulate samples");

        // Every quantity must be finite and physically plausible at the quiet start.
        for env in [latest, baseline] {
            assert!(env.tidal_strength.is_finite() && env.tidal_strength > 0.0);
            assert!(env.density.is_finite() && env.density > 0.0);
            assert!(
                (1.0..2000.0).contains(&env.dispersion_kms),
                "dispersion {} km/s implausible",
                env.dispersion_kms
            );
            assert!(env.encounter_rate.is_finite() && env.encounter_rate >= 0.0);
        }
    }
}
