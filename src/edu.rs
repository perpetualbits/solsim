//! Educational mode: a step-by-step, arrow-annotated walkthrough of how a
//! time-step gravity simulation works.
//!
//! It runs a tiny two-body demo (the Sun and one planet) that you advance one
//! small step at a time. Each step is broken into phases — see the velocity,
//! compute the gravitational pull, add the relativity correction, update the
//! velocity, then move — with big arrows for every vector and plain-language text.
//! Written for a 4-VWO reader: the maths is spelled out.

use glam::DVec3;

use crate::astro::constants::{C_LIGHT, GM_SUN};
use crate::render::arrows::ArrowInstance;
use crate::render::grid::LineSeg;
use crate::render::sphere::Instance;
use crate::render::textures;

/// How many explanation phases make up one step.
pub const PHASES: usize = 5;

/// Largest number of past points kept in the demo's path.
const MAX_PATH: usize = 4000;

/// Arrow shaft thickness, in AU.
const THICK: f32 = 0.012;
/// Extra visual exaggeration for the (tiny) GR arrow, so it can be seen.
const GR_ARROW_EXAG: f64 = 2000.0;
/// Cap on the GR arrow length, in AU.
const GR_ARROW_MAX: f64 = 0.22;

/// Drawn radii (AU) of the demo Sun and planet.
const SUN_RADIUS: f32 = 0.06;
const PLANET_RADIUS: f32 = 0.045;

/// The educational demo's state.
///
/// What: the planet's position and velocity, the step size, whether GR is on, the
/// path travelled, and which explanation phase we are in.
/// How/why: this is a self-contained two-body world (Sun at the origin) so the
/// student can watch one integration step at a time without the rest of the system
/// in the way.
/// Units: `r` in AU; `v` in AU/day; `dt` in days; `gr_strength` dimensionless.
pub struct Edu {
    pub r: DVec3,
    pub v: DVec3,
    pub dt: f64,
    pub gr: bool,
    pub gr_strength: f64,
    pub path: Vec<DVec3>,
    pub phase: usize,
    pub playing: bool,
    pub play_accum: f64,
}

impl Default for Edu {
    fn default() -> Self {
        let mut edu = Self {
            r: DVec3::ZERO,
            v: DVec3::ZERO,
            dt: 12.0,
            gr: false,
            gr_strength: 1.5e5,
            path: Vec::new(),
            phase: 0,
            playing: false,
            play_accum: 0.0,
        };
        edu.reset();
        edu
    }
}

impl Edu {
    /// Reset the demo to a fresh elliptical orbit.
    ///
    /// What: places the planet at 1.2 AU with a sub-circular sideways speed.
    /// How/why: a speed below the circular value gives a clear ellipse, so the
    /// motion (and later the GR precession) is easy to see; the path is cleared.
    /// Units: AU and AU/day.
    pub fn reset(&mut self) {
        self.r = DVec3::new(1.2, 0.0, 0.0);
        let v_circ = (GM_SUN / self.r.length()).sqrt();
        self.v = DVec3::new(0.0, v_circ * 0.85, 0.0);
        self.path.clear();
        self.path.push(self.r);
        self.phase = 0;
        self.play_accum = 0.0;
    }

    /// The gravitational and (scaled) GR accelerations at the current state.
    ///
    /// What: returns `(gravity, gr)` separately so the demo can show each arrow.
    /// How/why: gravity is `−GM·r/|r|³`; the GR term is the 1-post-Newtonian
    /// Schwarzschild correction, multiplied by `gr_strength` (exaggerated so its
    /// effect is visible within a few orbits).
    /// Units: AU/day².
    pub fn accel(&self) -> (DVec3, DVec3) {
        let r = self.r;
        let rl = r.length();
        let grav = -GM_SUN * r / (rl * rl * rl);
        let gr = if self.gr {
            let v = self.v;
            let v2 = v.length_squared();
            let rv = r.dot(v);
            let mu = GM_SUN;
            let c2 = C_LIGHT * C_LIGHT;
            self.gr_strength * (mu / (c2 * rl * rl * rl)) * ((4.0 * mu / rl - v2) * r + 4.0 * rv * v)
        } else {
            DVec3::ZERO
        };
        (grav, gr)
    }

    /// Perform one semi-implicit Euler step and record it.
    ///
    /// What: updates velocity then position by one `dt`.
    /// How/why: `v ← v + a·Δt`, then `r ← r + v·Δt` — the simplest stepping rule,
    /// the heart of the whole simulation. (Using the new velocity for the position
    /// keeps orbits stable.) The real program uses tiny RK4 steps for accuracy;
    /// here the step is big so each one is visible.
    /// Units: days for `dt`.
    pub fn commit_step(&mut self) {
        let (g, gr) = self.accel();
        self.v += (g + gr) * self.dt;
        self.r += self.v * self.dt;
        self.path.push(self.r);
        if self.path.len() > MAX_PATH {
            self.path.remove(0);
        }
    }

    /// Advance to the next phase, committing a step after the last one.
    ///
    /// What: steps the explanation forward; wraps from the final phase to the first
    /// while actually moving the planet.
    /// How/why: lets the student walk through compute → add → update → move, then
    /// take the step and repeat.
    /// Units: none.
    pub fn next(&mut self) {
        if self.phase + 1 >= PHASES {
            self.commit_step();
            self.phase = 0;
        } else {
            self.phase += 1;
        }
    }

    /// Auto-advance the walkthrough while "play" is on.
    ///
    /// What: ticks the phases forward on a timer.
    /// How/why: accumulates real time and advances one phase every ~0.6 s, so the
    /// demo can run itself.
    /// Units: `dt_real` in seconds.
    pub fn update(&mut self, dt_real: f64) {
        if !self.playing {
            return;
        }
        self.play_accum += dt_real;
        if self.play_accum >= 0.6 {
            self.play_accum = 0.0;
            self.next();
        }
    }

    /// The Sun and planet to draw, with the Sun textured and the planet (Earth).
    ///
    /// What: two [`Instance`]s for the demo.
    /// How/why: the Sun glows; the planet is lit and uses the Earth map.
    /// Units: positions in AU.
    pub fn instances(&self) -> Vec<Instance> {
        let sun = Instance {
            center: [0.0, 0.0, 0.0],
            radius: SUN_RADIUS,
            color: [1.0, 1.0, 1.0],
            emissive: 1.0,
            tex_layer: textures::layer_of("sun"),
            spin: [0.0, 0.0, 1.0, 0.0],
        };
        let p = self.r.as_vec3();
        let planet = Instance {
            center: [p.x, p.y, p.z],
            radius: PLANET_RADIUS,
            color: [1.0, 1.0, 1.0],
            emissive: 0.0,
            tex_layer: textures::layer_of("earth"),
            spin: [0.0, 0.0, 1.0, 0.0],
        };
        vec![sun, planet]
    }

    /// The path travelled so far, as faint line segments.
    ///
    /// What: connects the recorded points into a polyline.
    /// How/why: shows the orbit building up step by step (and, with GR on, the
    /// slow precession).
    /// Units: AU.
    pub fn path_segments(&self) -> Vec<LineSeg> {
        let color = [0.55, 0.6, 0.75, 0.8];
        self.path
            .windows(2)
            .map(|w| LineSeg {
                a: w[0],
                b: w[1],
                color,
                width: 1.6,
                fade: false,
            })
            .collect()
    }

    /// The arrows to show for the current phase.
    ///
    /// What: the vectors as big arrows (position, velocity, gravity, GR, new
    /// velocity), revealed cumulatively as the phases progress.
    /// How/why: velocity-type vectors are drawn scaled by `Δt` (so they show the
    /// *displacement per step* in AU) and acceleration-type by `Δt²` (their effect
    /// on position per step), so the arrows are directly comparable on the diagram.
    /// The GR arrow is additionally exaggerated and capped so it is visible.
    /// Units: arrow lengths in AU.
    pub fn arrows(&self) -> Vec<ArrowInstance> {
        let mut out = Vec::new();
        let r = self.r;
        let (g, gr) = self.accel();
        let dt = self.dt;

        let mut arrow = |start: DVec3, vec: DVec3, color: [f32; 4]| {
            let len = vec.length();
            if len < 1e-12 {
                return;
            }
            let d = (vec / len).as_vec3();
            let s = start.as_vec3();
            out.push(ArrowInstance {
                start: [s.x, s.y, s.z],
                dir: [d.x, d.y, d.z],
                length: len as f32,
                thickness: THICK,
                color,
            });
        };

        // Position vector r, Sun → planet (always).
        arrow(DVec3::ZERO, r, [1.0, 0.9, 0.2, 1.0]);
        // Velocity (displacement this step) v·Δt, from the planet (always).
        arrow(r, self.v * dt, [0.3, 1.0, 0.35, 1.0]);

        // Gravity's effect on position this step, g·Δt², from the planet.
        if self.phase >= 1 {
            arrow(r, g * dt * dt, [1.0, 0.35, 0.25, 1.0]);
        }
        // GR correction (exaggerated), from the planet.
        if self.phase >= 2 && self.gr {
            let mut grv = gr * dt * dt * GR_ARROW_EXAG;
            if grv.length() > GR_ARROW_MAX {
                grv = grv.normalize() * GR_ARROW_MAX;
            }
            arrow(r, grv, [1.0, 0.3, 1.0, 1.0]);
        }
        // The updated velocity (where it will head), from the planet.
        if self.phase >= 3 {
            let new_v = self.v + (g + gr) * dt;
            arrow(r, new_v * dt, [0.6, 1.0, 0.6, 1.0]);
        }

        out
    }

    /// The explanation text for the current phase, with live numbers.
    ///
    /// What: a short title and body describing what this phase does.
    /// How/why: spells out the formula and shows the current magnitudes so the
    /// student connects the arrows to the maths.
    /// Units: none (text).
    pub fn phase_text(&self) -> (String, String) {
        let rl = self.r.length();
        let vl = self.v.length();
        let (g, gr) = self.accel();
        match self.phase {
            0 => (
                "1. Where we are".into(),
                format!(
                    "The planet sits at distance r = {rl:.3} AU from the Sun (yellow arrow) \
and moves with velocity v = {vl:.4} AU/day (green arrow). The green arrow is drawn as \
v·Δt: how far it would drift in one time step Δt = {:.0} days if nothing pulled it.",
                    self.dt
                ),
            ),
            1 => (
                "2. Compute the pull (gravity)".into(),
                format!(
                    "Gravity pulls the planet straight at the Sun: a = −G·M·r / r³. \
Right now |a| = {:.2e} AU/day². The red arrow shows its effect on the path over one \
step (a·Δt²) — small, but it is what bends the straight green path into a curve.",
                    g.length()
                ),
            ),
            2 => {
                if self.gr {
                    (
                        "3. Add general relativity".into(),
                        format!(
                            "Einstein adds a tiny extra pull: a_GR = (μ/c²r³)·[(4μ/r − v²)·r + \
4(r·v)·v]. Here it is {:.2e} AU/day² (magenta arrow, greatly exaggerated to be \
visible). On its own it is far too small to notice in one step — but added every \
step it slowly turns the whole orbit (precession).",
                            (gr / self.gr_strength.max(1.0)).length()
                        ),
                    )
                } else {
                    (
                        "3. (General relativity is off)".into(),
                        "Turn on 'General relativity' to add Einstein's small extra pull. \
Without it, Newton's gravity alone gives a closed ellipse.".into(),
                    )
                }
            }
            3 => (
                "4. Update the velocity".into(),
                "New velocity = old velocity + acceleration × Δt:  v ← v + a·Δt. \
The brighter green arrow is the new velocity (again drawn ×Δt). Notice it has bent \
slightly toward the Sun compared with the old one — that is gravity at work."
                    .into(),
            ),
            _ => (
                "5. Move, then repeat".into(),
                "Finally move the planet: r ← r + v·Δt (using the new velocity). Press \
'Step' to take the step — the planet jumps to the arrow's tip and the path grows. \
Do it again and again and the planet traces its orbit. That is the whole simulation: \
compute the pull, nudge the velocity, move, repeat."
                    .into(),
            ),
        }
    }
}

/// Draw the educational control/explanation panel.
///
/// What: an egui window with the current step's text and the controls.
/// How/why: the buttons drive the walkthrough (next phase / take a step / play),
/// and the toggles let the student switch GR on and change the step size to see
/// its effect.
/// Units: none.
pub fn panel(ctx: &egui::Context, edu: &mut Edu) {
    egui::Window::new("Educational mode")
        .default_size([380.0, 360.0])
        .show(ctx, |ui| {
            let (title, body) = edu.phase_text();
            ui.heading(title);
            ui.label(body);
            ui.add_space(8.0);

            ui.horizontal(|ui| {
                if ui.button("▶ Next").clicked() {
                    edu.next();
                }
                if ui.button("⟳ Reset").clicked() {
                    edu.reset();
                }
                let play_label = if edu.playing { "⏸ Pause" } else { "⏵ Play" };
                if ui.button(play_label).clicked() {
                    edu.playing = !edu.playing;
                }
            });

            ui.add_space(4.0);
            ui.checkbox(&mut edu.gr, "General relativity (extra pull → precession)");
            ui.add(egui::Slider::new(&mut edu.dt, 2.0..=25.0).text("step size Δt (days)"));

            ui.separator();
            ui.label(
                "Arrows: yellow = position r, green = velocity (×Δt), red = gravity's \
bend (×Δt²), magenta = relativity (exaggerated). Lengths are scaled so they are \
comparable; the real program takes tiny RK4 steps instead of these big ones.",
            );

            ui.collapsing("Pseudocode", |ui| {
                ui.label("The whole simulation is this loop:");
                ui.monospace(PSEUDOCODE);
            });

            ui.label("Press K to leave educational mode.");
        });
}

/// The integration loop in plain pseudocode, for the manual/panel.
///
/// What: the time-step algorithm written out as readable steps.
/// How/why: shows students that the simulation is a short loop — compute the pull,
/// nudge the velocity, move, repeat — with the GR term as one extra line.
/// Units: none (text).
const PSEUDOCODE: &str = r#"# constants
G·M_sun = k²            # Sun's gravity strength
μ       = G·M_sun       # short name for it, used below
c       = 173.144       # speed of light, in AU/day
dt      = small step    # e.g. a fraction of a day

# starting point (from the real ephemeris)
r = position of planet
v = velocity of planet

repeat every step:
    # 1. acceleration from gravity (points at the Sun)
    a = -G·M_sun * r / |r|³

    # 2. add the general-relativity correction (tiny)
    a += (μ/(c²|r|³)) * ((4μ/|r| - v·v)*r + 4*(r·v)*v)

    # 3. update velocity, then position
    v = v + a * dt
    r = r + v * dt

    draw the planet at r
    advance the clock by dt

# the real program repeats this with RK4 (it samples the
# acceleration four times per step) for better accuracy."#;

#[cfg(test)]
mod tests {
    use super::*;

    /// The demo's semi-implicit Euler orbit must stay bounded (not spiral away).
    #[test]
    fn demo_orbit_is_bounded() {
        let mut e = Edu::default();
        for _ in 0..400 {
            e.commit_step();
            let rl = e.r.length();
            assert!((0.2..4.0).contains(&rl), "orbit left bounds at r = {rl}");
        }
    }

    /// More arrows are shown as the walkthrough progresses.
    #[test]
    fn arrows_grow_with_phase() {
        let mut e = Edu::default();
        e.gr = true;
        let early = e.arrows().len();
        e.phase = PHASES - 1;
        let late = e.arrows().len();
        assert!(late > early, "early {early} late {late}");
    }
}
