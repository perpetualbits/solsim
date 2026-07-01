//! Solar System Simulator — Phase 2: the Sun, Earth and Moon in 3-D.
//!
//! This file opens a Wayland window, sets up the GPU (wgpu), and each frame:
//! advances a clock, looks up the real positions of the Sun, Earth and Moon from
//! the ephemeris, and draws them as shaded spheres with fading trails, seen
//! through a mouse-controlled orbit camera. A small egui overlay shows the date,
//! the Earth–Sun distance and the controls.
//!
//! Why this structure: rendering on a GPU always needs the same four helpers —
//! an *instance* (the library entry point), a *surface* (the window we draw to),
//! a *device* (the GPU itself) and a *queue* (where we send drawing commands).
//! We build them once when the window appears, then reuse them every frame.

use std::sync::Arc;
use std::time::Instant;

use glam::DVec3;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

mod astro;
mod bodies;
mod config;
mod edu;
mod noise;
mod physics;
mod rng;
mod render;
mod stars;
mod ui;

use astro::time::SimClock;
use bodies::{BODIES, EARTH_INDEX};
use render::camera::OrbitCamera;
use render::grid::{self, LineSeg};
use render::sphere::Instance;
use render::starfield::StarInstance;
use render::trails::TrailSet;
use render::viewpoints::{self, Observer, Viewpoint};
use render::Scene;

/// Target spacing between trail samples, in simulated days.
///
/// What: how finely a frame's time step is cut up when sampling the trails.
/// How/why: at high speed a single frame can span many days, so we sample the
/// trails every ~this many days instead of once per frame; a few days keeps even
/// fast Mercury (88-day orbit) tracing a smooth ellipse rather than jagged chords.
/// Units: days.
const TRAIL_STEP_DAYS: f64 = 2.0;

/// Most trail sub-samples to take in a single frame.
///
/// What: a cap on the sub-stepping work per frame.
/// How/why: stops a huge time step (or a stutter) from costing thousands of
/// ephemeris evaluations; beyond this the trails coarsen but the program stays
/// responsive.
/// Units: a count.
const MAX_TRAIL_SUBSTEPS: u32 = 48;

/// How many energy samples the overlay graph keeps (its rolling window width).
///
/// What: the maximum length of the kinetic/potential/total-energy history.
/// How/why: one sample is recorded per simulated frame; keeping a few hundred
/// gives a readable scrolling graph without using noticeable memory.
/// Units: a count of samples.
const ENERGY_HIST_LEN: usize = 600;

/// Target integrator step, in simulated days.
///
/// What: how small a step the physics engine takes.
/// How/why: small steps keep the RK4 integration accurate, so orbits stay closed
/// (Newtonian) and the slow GR precession is faithful rather than numerical noise.
/// Units: days.
const PHYSICS_STEP_DAYS: f64 = 0.5;

/// Most integrator sub-steps to take in a single frame.
///
/// What: a cap on integration work per frame (the step *count*, not its size).
/// How/why: bounds the cost of a huge time step. Unlike before, the step *size* is
/// always held at [`PHYSICS_STEP_DAYS`] (see [`physics::nbody::plan_substeps`]), so
/// hitting this cap makes integrated time fall behind real time rather than
/// coarsen the step and corrupt short-period orbits. A bigger budget therefore
/// lets integrated time keep up to higher speeds before that happens: the maximum
/// integrated speed is about `budget · PHYSICS_STEP_DAYS · fps` simulated days per
/// real second — at 4096 · 0.5 · 60 ≈ 1.2×10⁵ days/s (~340 yr/s). Past that, up to
/// the 1×10⁸ days/s speed cap, integrated time deliberately lags and the HUD shows
/// a "speed-limited" note. Each sub-step is a cheap RK4 over ~9 bodies.
/// Units: a count.
const MAX_PHYSICS_SUBSTEPS: u32 = 4096;

/// Base colour of the adaptive 3-D grid (linear RGB; alpha is set per level).
const GRID_COLOR: [f32; 3] = [0.34, 0.40, 0.52];

/// Colour of the Earth-surface horizon line (linear RGBA).
const HORIZON_COLOR: [f32; 4] = [0.35, 0.55, 0.35, 0.65];

/// Thickness of the horizon line, in pixels.
const HORIZON_WIDTH: f32 = 1.6;

/// Default height of the Ecliptic-North camera above the Sun, in AU.
const DEFAULT_TOP_DISTANCE: f64 = 3.0;

/// View scale (characteristic distance, in AU) used for the Earth-surface grid.
const SURFACE_VIEW_SCALE: f64 = 1.0;

/// In logarithmic mode, how much to enlarge body draw radii (and the size limits),
/// so bodies stay visible at the compressed scale.
///
/// What: the size policy for log mode.
/// How/why: after distances are squashed, the bodies' true exaggerated radii are
/// tiny next to the compressed system, so we boost them by a constant factor and
/// clamp, keeping every body a visible dot with roughly its relative size.
/// Units: `BOOST` dimensionless; the clamps in display units.
const LOG_SIZE_BOOST: f32 = 6.0;
const LOG_MIN_SIZE: f32 = 0.005;
const LOG_MAX_SIZE: f32 = 0.12;

/// How much larger than the planet the cloud shell is drawn.
///
/// What: the cloud sphere's radius as a multiple of the body radius.
/// How/why: a thin shell just above the surface so clouds sit on the near side and
/// blend over the planet; ~2% keeps a clear depth gap without an obvious halo.
/// Units: dimensionless ratio.
const CLOUD_ALTITUDE: f32 = 1.02;

/// How much faster than the surface the clouds rotate.
///
/// What: the cloud layer's spin rate relative to the body's own rotation.
/// How/why: a few percent faster makes the clouds slowly slip eastward over the
/// surface (loosely, atmospheric super-rotation), so weather visibly drifts.
/// Units: dimensionless ratio.
const CLOUD_DRIFT: f64 = 1.03;

/// Closest the free camera may sit to the focused body, as a multiple of its
/// drawn radius.
///
/// What: the zoom-in stop, just above the surface.
/// How/why: a hair above the surface so you cannot fly through the body (and, with
/// the floating origin, fall toward its centre forever). It cannot be much smaller
/// because the near clip plane scales with the zoom — getting *much* closer than
/// this would clip the surface away anyway.
/// Units: dimensionless ratio.
const SURFACE_CLEARANCE: f64 = 1.005;

/// Decode the body texture maps into the layers of the GPU texture array.
///
/// What: builds one RGBA layer per body map, with a white layer 0 for untextured
/// bodies.
/// How/why: layer 0 is plain white so untextured bodies (small moons) show their
/// solid colour tint; the embedded maps follow in [`render::textures::TEXTURES`]
/// order, so `layer_of(name)` lines up with this list. The last layer is the
/// procedurally generated cloud map (`render::textures::cloud_layer`).
/// Units: each layer is `TEX_W·TEX_H·4` bytes of RGBA.
fn build_body_layers() -> Vec<Vec<u8>> {
    let white = vec![255u8; (render::textures::TEX_W * render::textures::TEX_H * 4) as usize];
    let mut layers = vec![white];
    for (_, bytes) in render::textures::TEXTURES {
        layers.push(render::textures::decode_rgba(bytes));
    }
    // The procedural cloud layer, baked once with fractal (fBm) noise.
    layers.push(render::clouds::bake_cloud_layer());
    layers
}

/// Load the star catalogue and turn it into ready-to-draw star instances.
///
/// What: builds the GPU star list — a direction, colour and size per star.
/// How/why: for each catalogue star we project (RA, Dec) to an ecliptic direction,
/// turn its B−V index into a colour, and its magnitude into a dot size. We then
/// append the faint procedural Milky Way band (stars concentrated along the
/// galactic plane). This is done once at start-up because the stars never change.
/// Units: directions are unit vectors; sizes in pixels; colours linear RGB.
fn build_star_instances() -> Vec<StarInstance> {
    let mut instances: Vec<StarInstance> = stars::catalog::load()
        .iter()
        .map(|s| {
            let dir = stars::project::radec_to_ecliptic(s.ra_deg, s.dec_deg);
            StarInstance {
                dir: [dir.x as f32, dir.y as f32, dir.z as f32],
                size: stars::color::magnitude_to_size(s.vmag),
                color: stars::color::bv_to_rgb(s.bv),
                _pad: 0.0,
            }
        })
        .collect();
    // The Milky Way band and the neighbour galaxies (Andromeda, the Magellanic
    // Clouds, …) — all faint stars on the same background (key `B` hides them too).
    instances.extend(stars::galaxy::milky_way_band());
    instances.extend(stars::galaxy::neighbor_galaxies());
    instances
}

/// The depth-buffer pixel format used for the whole program.
///
/// What: a constant naming how we store "how far away" each pixel is.
/// How: `Depth32Float` keeps one 32-bit floating-point distance per pixel; when
/// two triangles overlap, the GPU keeps the nearer one by comparing these values.
/// Principle: this is the standard "z-buffer" idea — closer things hide farther
/// things, just like in real life.
/// Units: a pure GPU format tag (no physical unit); the stored values are in the
/// normalised clip-space depth range 0..1.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Everything the GPU needs to draw into our window.
///
/// What: a bundle of the long-lived GPU objects plus the current window size.
/// How: created once in [`Gpu::new`] and then reused; on resize we only rebuild
/// the size-dependent parts (the surface configuration and the depth texture).
/// Principle: setting up a GPU is expensive, so we do it once and keep it.
/// Units: `config.width`/`config.height` are in physical pixels.
struct Gpu {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,
    screenshot_supported: bool,
}

impl Gpu {
    /// Build all GPU objects for a freshly created window.
    ///
    /// What: connects wgpu to the window and returns a ready-to-draw [`Gpu`].
    /// How: we (1) make an `Instance`, (2) make a `Surface` for the window,
    /// (3) ask for an `Adapter` (a real GPU that can draw to that surface),
    /// (4) ask the adapter for a `Device` + `Queue`, then (5) configure the
    /// surface and (6) make a matching depth texture. Steps 3 and 4 are
    /// asynchronous, so we block on them with `pollster`.
    /// Principle: a GPU pipeline is a chain — instance → surface → adapter →
    /// device → queue — where each link is needed to create the next.
    /// Units: window size in physical pixels; returns a `Result` because any step
    /// can fail on machines without a suitable GPU.
    fn new(window: Arc<Window>) -> Result<Self, Box<dyn std::error::Error>> {
        let size = window.inner_size();
        // A surface needs a non-zero size; clamp to at least 1×1 so configuration
        // never fails on a momentarily zero-sized window (can happen on Wayland).
        let width = size.width.max(1);
        let height = size.height.max(1);

        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(instance_desc);

        let surface = instance.create_surface(window.clone())?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))?;

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("solarsim device"),
                ..Default::default()
            }))?;

        // Choose a colour format the surface supports, preferring an sRGB one so
        // colours look correct on screen.
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        // Allow copying from the surface (for screenshots) when supported.
        let screenshot_supported = caps.usages.contains(wgpu::TextureUsages::COPY_SRC);
        let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        if screenshot_supported {
            usage |= wgpu::TextureUsages::COPY_SRC;
        }

        let config = wgpu::SurfaceConfiguration {
            usage,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let depth_view = create_depth_view(&device, width, height);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            screenshot_supported,
            depth_view,
        })
    }

    /// React to a window resize by rebuilding the size-dependent GPU state.
    ///
    /// What: reconfigures the surface and recreates the depth texture at the new
    /// size.
    /// How: store the new width/height in `config`, call `surface.configure`, then
    /// build a fresh depth texture of the same size; we ignore zero-sized resizes
    /// (which happen when a window is minimised).
    /// Principle: the surface and depth buffer must always match the window's pixel
    /// size, or drawing would be stretched or crash.
    /// Units: `width`/`height` in physical pixels.
    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = create_depth_view(&self.device, width, height);
    }
}

/// Create a depth texture and return a view we can attach when drawing.
///
/// What: makes the off-screen image that stores per-pixel distance.
/// How: allocate a texture of the given size in [`DEPTH_FORMAT`] with the
/// `RENDER_ATTACHMENT` usage (so the GPU may write depth to it), then return a
/// default view of it.
/// Principle: the depth ("z") buffer lets the GPU draw overlapping 3D shapes in
/// the correct front-to-back order.
/// Units: `width`/`height` in physical pixels.
fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// The egui overlay (immediate-mode GUI drawn on top of the 3D scene).
///
/// What: bundles the three pieces egui needs — the `Context` (UI state), the
/// winit input bridge (`State`), and the wgpu drawing back-end (`Renderer`).
/// How: egui rebuilds the whole UI every frame from code; `State` feeds it mouse
/// and keyboard events, and `Renderer` turns its shapes into GPU triangles.
/// Principle: "immediate mode" means there is no retained widget tree — the UI is
/// simply a function of the current state, re-run each frame.
/// Units: none (UI sizes are in logical points, scaled by the screen DPI).
struct EguiLayer {
    ctx: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

impl EguiLayer {
    /// Set up egui for a given window and surface colour format.
    ///
    /// What: builds the [`EguiLayer`] ready to receive events and draw.
    /// How: create an egui `Context`, wrap it in an `egui_winit::State` bound to
    /// the window, and make an `egui_wgpu::Renderer` that outputs in the same
    /// colour format as the surface (and with no depth buffer, since the overlay
    /// is drawn flat on top).
    /// Principle: keeping input handling (`State`) separate from drawing
    /// (`Renderer`) mirrors egui's design and keeps each part simple.
    /// Units: `max_texture_side` is in texels (the largest texture the GPU allows).
    fn new(window: &Window, device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let ctx = egui::Context::default();
        let max_texture_side = device.limits().max_texture_dimension_2d as usize;
        let state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            Some(max_texture_side),
        );
        let renderer =
            egui_wgpu::Renderer::new(device, color_format, egui_wgpu::RendererOptions::default());
        Self {
            ctx,
            state,
            renderer,
        }
    }
}

/// Which engine moves the bodies each frame.
///
/// What: the three ways the simulation can advance.
/// How/why: `Ephemeris` reads the real sky from formulas; `Newtonian` and
/// `Relativistic` integrate the equations of motion, the latter adding the GR
/// correction so orbits precess.
/// Units: none.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Engine {
    Ephemeris,
    Newtonian,
    Relativistic,
}

impl Engine {
    /// A short label for the HUD.
    fn name(self) -> &'static str {
        match self {
            Engine::Ephemeris => "Ephemeris",
            Engine::Newtonian => "Newtonian",
            Engine::Relativistic => "Relativistic (GR)",
        }
    }
}

/// Seed an integrator state from the analytic ephemeris at a given time.
///
/// What: builds the planets' starting positions and velocities for the physics
/// engine.
/// How/why: positions come straight from the ephemeris; velocities come from a
/// central finite difference of the ephemeris a fraction of a day either side, so
/// the integration starts off matching the real motion. This is what lets you flip
/// to Newtonian/GR mid-flight without the planets jumping.
/// Units: `jd` in days; returns a state with positions in AU and velocities in
/// AU/day.
fn seed_state(jd: f64) -> physics::nbody::State {
    let mut pos = Vec::with_capacity(bodies::PLANETS.len());
    let mut vel = Vec::with_capacity(bodies::PLANETS.len());
    for &planet in &bodies::PLANETS {
        pos.push(astro::ephemeris::planet_position(planet, jd));
        vel.push(astro::ephemeris::velocity_fd(
            |j| astro::ephemeris::planet_position(planet, j),
            jd,
            1.0 / 32.0,
        ));
    }
    physics::nbody::State { pos, vel }
}

/// The facts shown in the on-screen overlay this frame.
///
/// What: a small bundle of the numbers the HUD displays.
/// How/why: gathering them into one struct keeps the UI function tidy and free of
/// simulation details.
/// Units: `date` is text; `earth_sun_au` in AU; `speed_days_per_sec` in days per
/// real second; `fps` in frames per second.
struct HudInfo {
    date: String,
    earth_sun_au: f64,
    speed_factor: f64,
    physics_speed_limited: bool,
    paused: bool,
    fps: f32,
    viewpoint: &'static str,
    engine: &'static str,
    gr_strength: f64,
    show_gr: bool,
    log: bool,
    true_scale: bool,
    show_all: bool,
    show_grid: bool,
    show_stars: bool,
    show_help: bool,
}

/// Build the overlay panels shown each frame.
///
/// What: draws the status window (date, distance, speed, viewpoint) and, when
/// toggled on, a controls cheat-sheet.
/// How: egui's `Window` widgets lay out the labels; we reach the egui `Context`
/// through the root `Ui` we are handed.
/// Principle: immediate-mode UI — calling this every frame *is* the UI; there is
/// nothing to "update" separately.
/// Units: see [`HudInfo`].
fn build_overlay(ui: &mut egui::Ui, info: &HudInfo, focus: &mut usize) {
    let ctx = ui.ctx().clone();
    egui::Window::new("Solar System Simulator")
        .resizable(false)
        .show(&ctx, |ui| {
            // Focus picker: jump the Free view's centre to any body.
            egui::ComboBox::from_label("Focus")
                .selected_text(BODIES[(*focus).min(BODIES.len() - 1)].name)
                .show_ui(ui, |ui| {
                    for (i, body) in BODIES.iter().enumerate() {
                        ui.selectable_value(focus, i, body.name);
                    }
                });
            ui.separator();
            ui.label(format!("Date: {}", info.date));
            ui.label(format!("Earth–Sun: {:.4} AU", info.earth_sun_au));
            let speed = if info.paused {
                "paused".to_string()
            } else {
                format_speed(info.speed_factor)
            };
            ui.label(format!("Speed: {speed}"));
            // At extreme speed the integrated engines cannot keep up without
            // breaking short-period orbits, so time is held back on purpose; tell
            // the user why the clock is lagging.
            if info.physics_speed_limited {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 191, 0),
                    "⚠ physics speed-limited (step capped)",
                );
            }
            let view_label = if info.log {
                format!("{}  [LOG]", info.viewpoint)
            } else {
                info.viewpoint.to_string()
            };
            ui.label(format!("View: {view_label}"));
            let engine = if info.show_gr {
                format!("{}  ×{:.0}", info.engine, info.gr_strength)
            } else {
                info.engine.to_string()
            };
            ui.label(format!("Engine: {engine}"));
            let bodies = if info.show_all {
                "all planets & moons"
            } else {
                "Sun–Earth–Moon"
            };
            ui.label(format!("Bodies: {bodies}"));
            let grid = if info.show_grid { "on" } else { "off" };
            let stars = if info.show_stars { "on" } else { "off" };
            ui.label(format!("Grid: {grid}   Stars: {stars}"));
            let scale = if info.true_scale {
                "true (real radii)"
            } else {
                "exaggerated"
            };
            ui.label(format!("Body size: {scale}"));
            ui.label(format!("FPS: {:.0}", info.fps));
            ui.separator();
            ui.label("Press ? for controls");
        });

    if info.show_help {
        egui::Window::new("Controls")
            .resizable(false)
            .show(&ctx, |ui| {
                ui.label("Mouse drag — orbit camera (Free / Ecliptic-North)");
                ui.label("Mouse wheel — zoom");
                ui.label(".  /  ,  — time speed ×10 / ÷10");
                ui.label("Space — pause / resume time");
                ui.label("T — reset time to now");
                ui.label("V — cycle viewpoint");
                ui.label("Tab — focus next body (or pick in the HUD)");
                ui.label("P — show all planets & moons / just Sun–Earth–Moon");
                ui.label("C — toggle 3-D grid");
                ui.label("B — toggle star background");
                ui.label("R — clear trails");
                ui.label("L — toggle logarithmic distance mode");
                ui.label("S — body size: real ↔ exaggerated");
                ui.label("K — educational mode (vector walkthrough)");
                ui.label("Y — energy graph (kinetic / potential / total)");
                ui.label("J — Kepler equal-area sweep (focus a planet; top-down view)");
                ui.label("E / N / G — engine: Ephemeris / Newtonian / GR");
                ui.label("[  /  ]  — GR strength ÷10 / ×10");
                ui.label("F1 / H — open the manual");
                ui.label("F12 — save a screenshot (PNG)");
                ui.label("Q — quit the program");
                ui.label("?  — toggle this help");
            });
    }
}

/// Turn a Julian Date into a short calendar string like "2026-06-28".
///
/// What: formats the simulation date for the HUD.
/// How/why: converts the JD back to a calendar date with the astro module and
/// pads the month/day to two digits; if the conversion fails (only for absurd
/// dates) we fall back to printing the raw JD.
/// Units: `jd` in days; returns text.
fn format_date(jd: f64) -> String {
    match astro::time::calendar_from_jd(jd) {
        Ok((year, month, day)) => format!("{year:04}-{month:02}-{:02}", day as u32),
        Err(_) => format!("JD {jd:.2}"),
    }
}

/// Describe the time speed in human terms.
///
/// What: turns the raw speed factor into a readable rate like "10 days/s".
/// How/why: the factor is simulated seconds per real second; we convert it to the
/// largest natural unit (seconds → minutes → hours → days → years) so the number
/// stays small and meaningful at any speed.
/// Units: input is simulated seconds per real second; output is text.
fn format_speed(factor: f64) -> String {
    let days = factor / 86_400.0;
    if days >= 365.25 {
        format!("{:.1} yr/s", days / 365.25)
    } else if days >= 1.0 {
        format!("{days:.1} days/s")
    } else if days >= 1.0 / 24.0 {
        format!("{:.1} hr/s", days * 24.0)
    } else if days >= 1.0 / 1440.0 {
        format!("{:.1} min/s", days * 1440.0)
    } else {
        format!("{:.0} s/s", factor)
    }
}

/// The whole application: window, GPU, scene, simulation clock and input state.
///
/// What: holds everything the event loop needs between events.
/// How: the GPU-bound fields are `Option`s because winit only lets us create a
/// window after the app is "resumed"; they are `None` until then. The clock,
/// camera and trails exist from the start.
/// Principle: winit drives us through callbacks, so our state must live in a
/// struct that those callbacks share.
/// Units: `last_frame` is a moment in time; `fps` is in frames per second;
/// `last_cursor` is in physical pixels.
struct App {
    window: Option<Arc<Window>>,
    gpu: Option<Gpu>,
    egui: Option<EguiLayer>,
    scene: Option<Scene>,
    clock: SimClock,
    camera: OrbitCamera,
    trails: TrailSet,
    viewpoint: Viewpoint,
    observer: Observer,
    top_camera: OrbitCamera,
    focus_index: usize,
    engine: Engine,
    state: Option<physics::nbody::State>,
    gr_strength: f64,
    paused: bool,
    log: bool,
    true_scale: bool,
    show_all: bool,
    show_grid: bool,
    show_stars: bool,
    show_help: bool,
    show_manual: bool,
    manual_search: String,
    show_energy: bool,
    energy_hist: std::collections::VecDeque<ui::energy::Sample>,
    /// Whether the Kepler's-second-law equal-area sweep is shown for the focus.
    show_kepler: bool,
    /// True when the integrated engine could not keep up this frame and time was
    /// held back to protect orbit accuracy (shown as a HUD warning).
    physics_speed_limited: bool,
    edu: Option<edu::Edu>,
    screenshot_requested: bool,
    screenshot_count: u32,
    config: config::Config,
    dragging: bool,
    last_cursor: Option<(f64, f64)>,
    last_frame: Option<Instant>,
    fps: f32,
}

impl App {
    /// Create the application state (before the window exists).
    ///
    /// What: sets up the clock at "now", the default camera, and empty trails.
    /// How/why: the GPU-bound parts are filled in later (in `resumed`); here we
    /// only build the parts that need no GPU. The clock starts at the real current
    /// date running at [`DEFAULT_SPEED_FACTOR`], and one trail is made per body
    /// using that body's colour.
    /// Units: none.
    fn new(config: config::Config) -> Self {
        // Start at the configured date (or "now"), at the configured speed.
        let start_jd = match config.start_date {
            Some((y, m, d)) => astro::time::jd_from_calendar(y, m, d, 0, 0, 0.0),
            None => astro::time::jd_now(),
        };
        let mut clock = SimClock::new(start_jd);
        clock.set_speed_factor(config.speed_days_per_sec * 86_400.0);

        let colors: Vec<[f32; 3]> = BODIES.iter().map(|b| b.color).collect();

        // The top-down camera: an orbit camera centred on the Sun, starting nearly
        // straight down (φ ≈ 86°, clear of the pole) but free to be tilted/zoomed.
        let top_camera = OrbitCamera {
            target: DVec3::ZERO,
            radius: DEFAULT_TOP_DISTANCE,
            phi: 1.5,
            ..Default::default()
        };

        Self {
            window: None,
            gpu: None,
            egui: None,
            scene: None,
            clock,
            camera: OrbitCamera::default(),
            trails: TrailSet::new(&colors, config.trail_length),
            viewpoint: Viewpoint::Free,
            observer: Observer {
                lat_deg: config.observer_lat,
                lon_deg: config.observer_lon,
            },
            top_camera,
            focus_index: EARTH_INDEX,
            engine: Engine::Ephemeris,
            state: None,
            gr_strength: 1.0,
            paused: false,
            log: false,
            true_scale: false,
            show_all: false,
            show_grid: true,
            show_stars: true,
            show_help: false,
            show_manual: false,
            manual_search: String::new(),
            show_energy: false,
            energy_hist: std::collections::VecDeque::new(),
            show_kepler: false,
            physics_speed_limited: false,
            edu: None,
            screenshot_requested: false,
            screenshot_count: 0,
            config,
            dragging: false,
            last_cursor: None,
            last_frame: None,
            fps: 0.0,
        }
    }
    /// Multiply the simulation speed, keeping it in a sensible range.
    ///
    /// What: scales how fast simulated time runs.
    /// How/why: the `.` and `,` keys call this with ×10 and ÷10; we clamp the
    /// result so time never stops or runs absurdly fast. Slowing down is what makes
    /// the spinning Earth-surface view watchable.
    /// Units: `factor` is dimensionless; the clamp range is in simulated seconds
    /// per real second.
    fn change_speed(&mut self, factor: f64) {
        let new_speed = (self.clock.speed_factor() * factor).clamp(1.0, 1.0e8);
        self.clock.set_speed_factor(new_speed);
    }

    /// Move the camera focus to the next/previous visible body.
    ///
    /// What: changes which body the Free view is centred on.
    /// How/why: `Tab` calls this; we step through the body list, skipping bodies
    /// that are currently hidden (so it cycles only what you can see), and wrap
    /// around. The combo box in the HUD can jump to any body directly.
    /// Units: none.
    fn cycle_focus(&mut self, forward: bool) {
        let n = BODIES.len();
        let mut i = self.focus_index;
        for _ in 0..n {
            i = if forward {
                (i + 1) % n
            } else {
                (i + n - 1) % n
            };
            if self.show_all || BODIES[i].core {
                self.focus_index = i;
                break;
            }
        }
    }

    /// Draw one frame: advance the simulation, render the scene, draw the HUD.
    ///
    /// What: produces a single rendered image and presents it to the window.
    /// How: (1) measure the elapsed time and advance the clock, (2) look up the
    /// bodies' positions and update the camera target and trails, (3) build the
    /// per-body instances and trail vertices, (4) lay out the egui HUD, (5) clear
    /// and draw the 3-D scene, then the HUD on top, and (6) present the frame.
    /// Principle: each frame is built from scratch — update, draw, present — the
    /// standard real-time loop.
    /// Units: none. If the surface image cannot be acquired (e.g. the window was
    /// just resized) we reconfigure or skip this frame and try again next time.
    fn draw(&mut self) {
        // --- frame timing: smoothed FPS, and the real time step to advance ---
        let now = Instant::now();
        let dt = self
            .last_frame
            .map(|p| now.duration_since(p).as_secs_f64())
            // Clamp so a stall (e.g. window drag) cannot teleport the simulation.
            .map(|d| d.min(0.1))
            .unwrap_or(0.0);
        self.last_frame = Some(now);
        if dt > 0.0 {
            let instant_fps = (1.0 / dt) as f32;
            self.fps = if self.fps == 0.0 {
                instant_fps
            } else {
                self.fps * 0.9 + instant_fps * 0.1
            };
        }

        // Pull the screenshot request out now, before the GPU fields are borrowed.
        let want_screenshot = std::mem::take(&mut self.screenshot_requested);
        let shot_index = self.screenshot_count;
        if want_screenshot {
            self.screenshot_count += 1;
        }

        // --- advance time, sampling trails finely across the step ------------
        // While paused, time stands still. Otherwise we cut this frame's (possibly
        // huge) time step into small sim-day increments and record a trail sample
        // at each, so fast inner planets trace smooth orbits instead of jagged,
        // skipped chords (temporal aliasing) at high speed.
        let jd_start = self.clock.jd();
        if !self.paused {
            self.clock.advance(dt);
        }
        // `jd` is the time we will draw at. In the integrated engines it can be
        // rolled back below if the physics could not keep up this frame.
        let mut jd = self.clock.jd();
        let dt_days = jd - jd_start;

        let positions = match self.engine {
            // Analytic ephemeris: sample the real sky directly at sub-steps.
            Engine::Ephemeris => {
                // The ephemeris is exact at any time, so it never speed-limits.
                self.physics_speed_limited = false;
                if dt_days != 0.0 {
                    let n = (dt_days.abs() / TRAIL_STEP_DAYS)
                        .ceil()
                        .clamp(1.0, MAX_TRAIL_SUBSTEPS as f64) as u32;
                    let mut latest = None;
                    for k in 1..=n {
                        let jd_k = jd_start + dt_days * (k as f64 / n as f64);
                        let p = bodies::system_positions(jd_k);
                        self.trails.record(&p);
                        latest = Some(p);
                    }
                    latest.unwrap_or_else(|| bodies::system_positions(jd))
                } else {
                    bodies::system_positions(jd)
                }
            }
            // Numerical integration: Newtonian gravity, optionally with the GR term.
            Engine::Newtonian | Engine::Relativistic => {
                let gr = if self.engine == Engine::Relativistic {
                    self.gr_strength
                } else {
                    0.0
                };
                // Take the state out so we can also borrow the trails.
                if let Some(mut state) = self.state.take() {
                    let mut limited = false;
                    if dt_days != 0.0 {
                        // Bound the step *size* (not just the count) so short-period
                        // orbits stay accurate no matter how fast time is requested.
                        let (n, h, lim) = physics::nbody::plan_substeps(
                            dt_days,
                            PHYSICS_STEP_DAYS,
                            MAX_PHYSICS_SUBSTEPS,
                        );
                        for k in 1..=n {
                            physics::nbody::rk4_step(
                                &mut state,
                                &bodies::PLANET_GM,
                                astro::constants::GM_SUN,
                                gr,
                                h,
                            );
                            let jd_k = jd_start + h * k as f64;
                            self.trails.record(&bodies::assemble(jd_k, &state.pos));
                        }
                        if lim {
                            // We only advanced by n·h (< dt_days). Roll the clock
                            // back to the time the physics actually reached, so the
                            // sim falls behind real time instead of coarsening.
                            let actual = jd_start + h * n as f64;
                            self.clock.set_jd(actual);
                            jd = actual;
                        }
                        limited = lim;
                    }
                    self.physics_speed_limited = limited;
                    let pos = bodies::assemble(jd, &state.pos);
                    self.state = Some(state);
                    pos
                } else {
                    self.physics_speed_limited = false;
                    bodies::system_positions(jd)
                }
            }
        };

        // Record the system's energy for the overlay graph, but only while time is
        // actually advancing (so a paused sim freezes the graph) and only when the
        // graph is open (the ephemeris path needs extra velocity evaluations). The
        // planets are BODIES indices 1..=8, matching the integrator's PLANET_GM and,
        // in N-body mode, the integrator's own velocities.
        if self.show_energy && dt_days != 0.0 {
            let planet_pos = &positions[1..=bodies::PLANETS.len()];
            let vel: Vec<DVec3> = match &self.state {
                Some(st) => st.vel.clone(),
                None => bodies::PLANETS
                    .iter()
                    .map(|&p| {
                        astro::ephemeris::velocity_fd(
                            |j| astro::ephemeris::planet_position(p, j),
                            jd,
                            1.0 / 32.0,
                        )
                    })
                    .collect(),
            };
            let (ke, pe) = physics::energy::system_energy(
                planet_pos,
                &vel,
                &bodies::PLANET_GM,
                astro::constants::GM_SUN,
            );
            self.energy_hist.push_back(ui::energy::Sample {
                ke,
                pe,
                total: ke + pe,
            });
            while self.energy_hist.len() > ENERGY_HIST_LEN {
                self.energy_hist.pop_front();
            }
        }

        let earth = positions[EARTH_INDEX];
        let focus = positions[self.focus_index.min(positions.len() - 1)];

        // The free camera is centred on the chosen focus body.
        self.camera.target = focus;

        // Stop the zoom just above the focused body's surface, so you cannot fly
        // through it and fall toward its centre forever. The "surface" is whatever
        // size the body is drawn at right now (exaggerated, true-scale, or log).
        let focus_i = self.focus_index.min(BODIES.len() - 1);
        let focus_surface = if self.log {
            (BODIES[focus_i].draw_radius_au * LOG_SIZE_BOOST).clamp(LOG_MIN_SIZE, LOG_MAX_SIZE)
                as f64
        } else if self.true_scale {
            bodies::real_radius_au(BODIES[focus_i].name)
        } else {
            BODIES[focus_i].draw_radius_au as f64
        };
        self.camera.min_radius = focus_surface * SURFACE_CLEARANCE;
        self.camera.radius = self.camera.radius.max(self.camera.min_radius);
        // The top-down view orbits the Sun; keep it from diving into the Sun too.
        let sun_surface = if self.true_scale {
            bodies::real_radius_au("Sun")
        } else {
            BODIES[0].draw_radius_au as f64
        };
        self.top_camera.min_radius = sun_surface * SURFACE_CLEARANCE;
        self.top_camera.radius = self.top_camera.radius.max(self.top_camera.min_radius);

        // The floating-origin centre depends on the viewpoint: the top-down view
        // is centred on the Sun, the surface view on the Earth (the observer), and
        // the free view on the chosen focus body.
        let origin = match self.viewpoint {
            Viewpoint::EclipticNorth => DVec3::ZERO,
            Viewpoint::EarthSurface => earth,
            Viewpoint::Free => focus,
        };
        // The observer's local sky axes (used by the Earth-surface view/horizon).
        let (zenith, north, east) = viewpoints::local_sky_basis(jd, &self.observer);
        // In the surface view we look at the southern sky, tilted ~35° up so the
        // sky fills the view rather than the horizon.
        let surface_forward = (-north * 0.819 + zenith * 0.574).normalize();
        let surface = self.viewpoint == Viewpoint::EarthSurface;

        // Which bodies to show: the Sun–Earth–Moon "core" always, the rest only
        // when P has turned on the full system. In the surface view we hide the
        // Earth itself — the observer is standing on it, so drawing it would just
        // wrap the camera inside the planet.
        let visible: Vec<bool> = BODIES
            .iter()
            .enumerate()
            .map(|(i, b)| (self.show_all || b.core) && !(surface && i == EARTH_INDEX))
            .collect();

        // The display transform: identity normally, the logarithmic squash in log
        // mode. It is applied only here, at draw time — the stored positions and
        // physics are never changed.
        let log = self.log;
        let display = |p: DVec3| {
            if log {
                render::logscale::compress(p)
            } else {
                p
            }
        };
        let origin_disp = display(origin);

        // --- build the GPU data for this frame -------------------------------
        let mut instances: Vec<Instance> = BODIES
            .iter()
            .zip(positions.iter())
            .zip(visible.iter())
            .filter(|(_, &vis)| vis)
            .map(|((spec, pos), _)| {
                let rel = (display(*pos) - origin_disp).as_vec3();
                // In log mode, distances shrink, so bodies are boosted (and
                // clamped) to stay visible at the compressed scale. In true-scale
                // mode the real (tiny) radii are used instead of the exaggerated
                // ones.
                let radius = if log {
                    (spec.draw_radius_au * LOG_SIZE_BOOST).clamp(LOG_MIN_SIZE, LOG_MAX_SIZE)
                } else if self.true_scale {
                    bodies::real_radius_au(spec.name) as f32
                } else {
                    spec.draw_radius_au
                };
                // Bodies with a texture map sample it (white tint); the rest use
                // the white layer 0 and show their solid colour.
                let tex_layer = render::textures::layer_of(&spec.name.to_lowercase());
                let color = if tex_layer != 0 {
                    [1.0, 1.0, 1.0]
                } else {
                    spec.color
                };

                // Axial spin: rotate the texture about an axis tilted by the
                // body's obliquity, by an angle that grows with time.
                let (period, obliquity_deg) = bodies::rotation(spec.name);
                let spin = if period != 0.0 {
                    let obl = obliquity_deg.to_radians();
                    let axis = DVec3::new(0.0, obl.sin(), obl.cos()).normalize();
                    let angle = (std::f64::consts::TAU * (jd - astro::time::J2000) / period)
                        .rem_euclid(std::f64::consts::TAU);
                    [axis.x as f32, axis.y as f32, axis.z as f32, angle as f32]
                } else {
                    [0.0, 0.0, 1.0, 0.0]
                };

                Instance {
                    center: [rel.x, rel.y, rel.z],
                    radius,
                    color,
                    emissive: if spec.emissive { 1.0 } else { 0.0 },
                    tex_layer,
                    spin,
                }
            })
            .collect();

        // A translucent cloud shell over the Earth: a slightly larger sphere with
        // the procedural fBm cloud map, rotating a touch faster than the surface so
        // the weather visibly drifts. Skipped in log mode (which would distort it)
        // and whenever the Earth itself is hidden (e.g. the surface view).
        let mut cloud_instances: Vec<Instance> = Vec::new();
        if !log && visible[EARTH_INDEX] {
            let rel = (display(positions[EARTH_INDEX]) - origin_disp).as_vec3();
            let earth_radius = if self.true_scale {
                bodies::real_radius_au("Earth") as f32
            } else {
                BODIES[EARTH_INDEX].draw_radius_au
            };
            let (period, obliquity_deg) = bodies::rotation("Earth");
            let spin = if period != 0.0 {
                let obl = obliquity_deg.to_radians();
                let axis = DVec3::new(0.0, obl.sin(), obl.cos()).normalize();
                let angle = (std::f64::consts::TAU * (jd - astro::time::J2000)
                    / (period / CLOUD_DRIFT))
                    .rem_euclid(std::f64::consts::TAU);
                [axis.x as f32, axis.y as f32, axis.z as f32, angle as f32]
            } else {
                [0.0, 0.0, 1.0, 0.0]
            };
            cloud_instances.push(Instance {
                center: [rel.x, rel.y, rel.z],
                radius: earth_radius * CLOUD_ALTITUDE,
                color: [1.0, 1.0, 1.0],
                emissive: 0.0,
                tex_layer: render::textures::cloud_layer(),
                spin,
            });
        }

        let (mut trail_vertices, mut trail_ranges) = self.trails.build(display, origin, &visible);
        let mut sun_pos = (DVec3::ZERO - origin_disp).as_vec3();

        // How "zoomed in" each view is, used to pick the grid's level of detail.
        let view_scale = match self.viewpoint {
            Viewpoint::Free => self.camera.radius,
            Viewpoint::EclipticNorth => self.top_camera.radius,
            Viewpoint::EarthSurface => SURFACE_VIEW_SCALE,
        };

        // Build the adaptive 3-D grid (any view) and the horizon (surface only) in
        // world coordinates, then shift them into the floating-origin frame. The
        // grid and horizon are hidden in log mode (they are linear-space tools; the
        // compressed orbits serve as the distance reference instead).
        let mut line_segs: Vec<LineSeg> = Vec::new();
        if self.show_grid && !log {
            line_segs.extend(grid::build_grid(origin, view_scale, GRID_COLOR));
        }
        if self.viewpoint == Viewpoint::EarthSurface && !log {
            let pairs = viewpoints::horizon_segments(earth, east, north);
            for chunk in pairs.chunks_exact(2) {
                line_segs.push(LineSeg {
                    a: chunk[0],
                    b: chunk[1],
                    color: HORIZON_COLOR,
                    width: HORIZON_WIDTH,
                    fade: false,
                });
            }
        }
        // Kepler's second law: shade the equal-area sectors that the Sun–planet
        // line sweeps in equal time intervals over one orbit of the focused planet.
        // Skipped in log mode and unless a planet (not the Sun or a moon) is focused.
        let mut area_verts: Vec<render::areas::AreaVertex> = Vec::new();
        if self.show_kepler && !log {
            let fi = self.focus_index;
            if (1..=bodies::PLANETS.len()).contains(&fi) {
                let planet = bodies::PLANETS[fi - 1];
                let r = positions[fi];
                let v = astro::ephemeris::velocity_fd(
                    |t| astro::ephemeris::planet_position(planet, t),
                    jd,
                    1.0 / 32.0,
                );
                let mu = astro::constants::GM_SUN;
                // Semi-major axis from the vis-viva relation, then the period.
                let a = 1.0 / (2.0 / r.length() - v.length_squared() / mu);
                if a.is_finite() && a > 0.0 {
                    let period = std::f64::consts::TAU * (a * a * a / mu).sqrt();
                    const SECTORS: usize = 12;
                    const SUB: usize = 10; // arc steps per sector
                    let total = SECTORS * SUB;
                    let sample = |k: usize| {
                        astro::ephemeris::planet_position(
                            planet,
                            jd + period * (k as f64) / (total as f64),
                        )
                    };
                    let to_v = |p: DVec3, c: [f32; 4]| {
                        let q = (p - origin).as_vec3();
                        render::areas::AreaVertex {
                            pos: [q.x, q.y, q.z],
                            color: c,
                        }
                    };
                    // Alternating shades make neighbouring (equal-area) sectors stand
                    // apart; the Sun sits at the heliocentric origin.
                    let fill_a = [0.30, 0.55, 0.95, 0.16];
                    let fill_b = [0.95, 0.65, 0.25, 0.16];
                    let orbit_col = [0.55, 0.65, 0.85, 0.8];
                    let spoke_col = [1.0, 0.85, 0.35, 0.9];
                    for s in 0..SECTORS {
                        let fill = if s % 2 == 0 { fill_a } else { fill_b };
                        for j in 0..SUB {
                            let p0 = sample(s * SUB + j);
                            let p1 = sample(s * SUB + j + 1);
                            // Triangle Sun–p0–p1 fills a thin slice of the wedge.
                            area_verts.push(to_v(DVec3::ZERO, fill));
                            area_verts.push(to_v(p0, fill));
                            area_verts.push(to_v(p1, fill));
                            // The orbit arc itself.
                            line_segs.push(LineSeg {
                                a: p0,
                                b: p1,
                                color: orbit_col,
                                width: 1.5,
                                fade: false,
                            });
                        }
                        // The radius vector (spoke) at the start of this sector.
                        line_segs.push(LineSeg {
                            a: DVec3::ZERO,
                            b: sample(s * SUB),
                            color: spoke_col,
                            width: 1.5,
                            fade: false,
                        });
                    }
                }
            }
        }

        for s in &mut line_segs {
            s.a -= origin;
            s.b -= origin;
        }
        let grid_fade = grid::fade_distances(view_scale);

        // Saturn's rings, when Saturn is shown (and not in log mode, which would
        // distort them). Built around Saturn's position in the render frame.
        let mut ring_verts = if !log && visible[bodies::SATURN_INDEX] {
            let center = positions[bodies::SATURN_INDEX] - origin;
            let saturn_radius = if self.true_scale {
                bodies::real_radius_au(BODIES[bodies::SATURN_INDEX].name)
            } else {
                BODIES[bodies::SATURN_INDEX].draw_radius_au as f64
            };
            render::rings::build(center, saturn_radius)
        } else {
            Vec::new()
        };

        let info = HudInfo {
            date: format_date(jd),
            earth_sun_au: earth.length(),
            speed_factor: self.clock.speed_factor(),
            physics_speed_limited: self.physics_speed_limited,
            paused: self.paused,
            fps: self.fps,
            viewpoint: self.viewpoint.name(),
            engine: self.engine.name(),
            gr_strength: self.gr_strength,
            show_gr: self.engine == Engine::Relativistic,
            log: self.log,
            true_scale: self.true_scale,
            show_all: self.show_all,
            show_grid: self.show_grid,
            show_stars: self.show_stars,
            show_help: self.show_help,
        };

        // --- render (needs the GPU-bound parts) ------------------------------
        let (Some(window), Some(gpu), Some(egui), Some(scene)) = (
            self.window.as_ref(),
            self.gpu.as_mut(),
            self.egui.as_mut(),
            self.scene.as_ref(),
        ) else {
            return;
        };

        let aspect = gpu.config.width as f32 / gpu.config.height.max(1) as f32;
        let mut view_proj = match self.viewpoint {
            Viewpoint::Free => self.camera.view_proj(aspect),
            Viewpoint::EclipticNorth => self.top_camera.view_proj(aspect),
            // Look at the southern sky (tilted up) with the local zenith as "up".
            Viewpoint::EarthSurface => {
                viewpoints::earth_surface_view_proj(surface_forward, zenith, aspect)
            }
        };
        // The stars use a rotation-only camera so they stay fixed on the sky. The
        // Earth-surface view is already at the origin, so its matrix works as-is.
        let mut star_view_proj = match self.viewpoint {
            Viewpoint::Free => self.camera.star_view_proj(aspect),
            Viewpoint::EclipticNorth => self.top_camera.star_view_proj(aspect),
            Viewpoint::EarthSurface => {
                viewpoints::earth_surface_view_proj(surface_forward, zenith, aspect)
            }
        };
        let mut arrows: Vec<render::arrows::ArrowInstance> = Vec::new();

        // Run the egui UI for this frame. The focus picker, manual and educational
        // panel write into local copies, applied back to `self` afterwards.
        let mut new_focus = self.focus_index;
        let mut manual_open = self.show_manual;
        let mut manual_search = std::mem::take(&mut self.manual_search);
        let mut energy_open = self.show_energy;
        // The graph reads a flat copy of the history (a VecDeque is not contiguous).
        let energy_samples: Vec<ui::energy::Sample> = self.energy_hist.iter().copied().collect();
        let mut edu_taken = self.edu.take();
        let raw_input = egui.state.take_egui_input(window);
        let full_output = egui.ctx.run_ui(raw_input, |ui| {
            if let Some(e) = edu_taken.as_mut() {
                edu::panel(ui.ctx(), e);
            } else {
                build_overlay(ui, &info, &mut new_focus);
                ui::manual::show(ui.ctx(), &mut manual_open, &mut manual_search);
                ui::energy::show(ui.ctx(), &mut energy_open, &energy_samples);
            }
        });
        self.edu = edu_taken;
        self.show_energy = energy_open;

        // In educational mode, replace the scene with the two-body demo: the Sun
        // and one planet, big vector arrows, and the path it has traced. The live
        // system keeps ticking in the background but is not shown.
        if let Some(e) = self.edu.as_mut() {
            e.update(dt);
            view_proj = self.camera.view_proj(aspect);
            star_view_proj = self.camera.star_view_proj(aspect);
            sun_pos = glam::Vec3::ZERO;
            instances = e.instances();
            trail_vertices = Vec::new();
            trail_ranges = Vec::new();
            line_segs = e.path_segments();
            ring_verts = Vec::new();
            cloud_instances = Vec::new();
            area_verts = Vec::new();
            arrows = e.arrows();
        }
        egui.state
            .handle_platform_output(window, full_output.platform_output);
        let paint_jobs = egui
            .ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [gpu.config.width, gpu.config.height],
            pixels_per_point: full_output.pixels_per_point,
        };

        // Acquire the image we will draw into.
        let frame = match gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                let (w, h) = (gpu.config.width, gpu.config.height);
                gpu.resize(w, h);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
            wgpu::CurrentSurfaceTexture::Validation => {
                eprintln!("surface validation error while acquiring frame");
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame encoder"),
            });

        // Upload any new/changed egui textures (e.g. the font atlas) first.
        for (id, delta) in &full_output.textures_delta.set {
            egui.renderer
                .update_texture(&gpu.device, &gpu.queue, *id, delta);
        }
        // Upload egui's vertex/index buffers; this may produce extra commands we
        // must submit before our own. (Must happen outside any render pass.)
        let egui_cmds = egui.renderer.update_buffers(
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            &paint_jobs,
            &screen,
        );

        // Pass 1: clear and draw the 3-D scene (trails, then bodies).
        scene.render(
            &gpu.queue,
            &mut encoder,
            &view,
            &gpu.depth_view,
            view_proj,
            sun_pos,
            [gpu.config.width as f32, gpu.config.height as f32],
            grid_fade,
            star_view_proj,
            self.show_stars,
            &instances,
            &cloud_instances,
            &trail_vertices,
            &trail_ranges,
            &line_segs,
            &ring_verts,
            &area_verts,
            &arrows,
        );

        // Pass 2: draw the egui overlay on top, keeping the rendered scene.
        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // egui's renderer needs a 'static render pass, so we drop the borrow
            // tie with `forget_lifetime` (safe: the pass lives only in this block).
            let mut render_pass = render_pass.forget_lifetime();
            egui.renderer.render(&mut render_pass, &paint_jobs, &screen);
        }

        // Submit egui's buffer-upload commands, then our drawing commands.
        gpu.queue.submit(
            egui_cmds
                .into_iter()
                .chain(std::iter::once(encoder.finish())),
        );

        // Save a screenshot of the rendered frame before presenting it.
        if want_screenshot {
            if gpu.screenshot_supported {
                let path = std::path::PathBuf::from(format!("solarsim-{shot_index:04}.png"));
                match render::screenshot::capture(
                    &gpu.device,
                    &gpu.queue,
                    &frame.texture,
                    gpu.config.width,
                    gpu.config.height,
                    gpu.config.format,
                    &path,
                ) {
                    Ok(()) => println!("Saved screenshot to {}", path.display()),
                    Err(e) => eprintln!("screenshot failed: {e}"),
                }
            } else {
                eprintln!("screenshots are not supported by this display surface");
            }
        }

        frame.present();

        // Free any egui textures that are no longer needed.
        for id in &full_output.textures_delta.free {
            egui.renderer.free_texture(id);
        }

        // Apply UI state chosen this frame (the egui borrow has now ended).
        self.show_manual = manual_open;
        self.manual_search = manual_search;
        // Picking a non-core body also reveals the full system so it is visible.
        self.focus_index = new_focus.min(BODIES.len() - 1);
        if !BODIES[self.focus_index].core {
            self.show_all = true;
        }
    }
}

impl ApplicationHandler for App {
    /// Create the window and GPU when the app becomes active.
    ///
    /// What: winit calls this when the platform is ready; we build the window,
    /// the GPU and the egui overlay here.
    /// How: ask the event loop for a resizable window, wrap it in an `Arc` so the
    /// GPU surface can keep it alive, then build [`Gpu`] and [`EguiLayer`]. If any
    /// step fails we print the error and ask the loop to exit cleanly.
    /// Principle: on Wayland a window may only exist while the app is "resumed",
    /// so creation must happen in this callback, not in `main`.
    /// Units: none.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return; // already set up
        }

        let attributes = Window::default_attributes()
            .with_title("Solar System Simulator")
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.config.window_width,
                self.config.window_height,
            ));
        let window = match event_loop.create_window(attributes) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("failed to create window: {e}");
                event_loop.exit();
                return;
            }
        };

        let gpu = match Gpu::new(window.clone()) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("failed to initialise GPU: {e}");
                event_loop.exit();
                return;
            }
        };

        let egui = EguiLayer::new(&window, &gpu.device, gpu.config.format);

        let scene = Scene::new(
            &gpu.device,
            &gpu.queue,
            gpu.config.format,
            DEPTH_FORMAT,
            BODIES.len() as u32,
            self.config.trail_length as u32,
            &build_body_layers(),
            &build_star_instances(),
        );

        self.egui = Some(egui);
        self.scene = Some(scene);
        self.gpu = Some(gpu);
        self.window = Some(window);
    }

    /// Handle one window event (resize, close, redraw, input, …).
    ///
    /// What: routes each event to egui and to our own resize/close/draw logic.
    /// How: first let egui see the event (so it can handle clicks on the overlay),
    /// then match on the events we care about; `RedrawRequested` is where we draw.
    /// On a lost/outdated surface we reconfigure and try again next frame.
    /// Principle: a GUI app is event-driven — it sleeps until the OS sends an
    /// event, then reacts.
    /// Units: `WindowEvent::Resized` carries a size in physical pixels.
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Let egui consume input first (e.g. dragging the overlay window). If egui
        // used the event, we do not also act on it for the camera.
        let mut egui_consumed = false;
        if let (Some(window), Some(egui)) = (self.window.as_ref(), self.egui.as_mut()) {
            egui_consumed = egui.state.on_window_event(window, &event).consumed;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => self.draw(),

            // Left button down/up toggles "drag to orbit" mode.
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.dragging = state == ElementState::Pressed && !egui_consumed;
            }

            // Moving the mouse while dragging rotates the active orbit camera.
            WindowEvent::CursorMoved { position, .. } => {
                let pos = (position.x, position.y);
                if let Some((px, py)) = self.last_cursor {
                    if self.dragging && !egui_consumed {
                        let dx = pos.0 - px;
                        let dy = pos.1 - py;
                        // Scale pixels to radians; drag right → spin left, drag up
                        // → tilt up (hence the sign choices).
                        match self.viewpoint {
                            Viewpoint::Free => self.camera.orbit(-dx * 0.005, dy * 0.005),
                            Viewpoint::EclipticNorth => {
                                self.top_camera.orbit(-dx * 0.005, dy * 0.005)
                            }
                            // The sky view's orientation is fixed by sidereal time.
                            Viewpoint::EarthSurface => {}
                        }
                    }
                }
                self.last_cursor = Some(pos);
            }

            // The scroll wheel zooms the active orbit camera.
            WindowEvent::MouseWheel { delta, .. } if !egui_consumed => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 50.0,
                };
                // Each step changes the distance by 10%; scroll up = closer.
                let factor = 0.9f64.powf(steps);
                match self.viewpoint {
                    Viewpoint::Free => self.camera.zoom(factor),
                    Viewpoint::EclipticNorth => self.top_camera.zoom(factor),
                    // The sky view has a fixed field of view.
                    Viewpoint::EarthSurface => {}
                }
            }

            // Keyboard shortcuts.
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                match &event.logical_key {
                    Key::Character(s) => match s.to_lowercase().as_str() {
                        "v" => self.viewpoint = self.viewpoint.next(),
                        "c" => self.show_grid = !self.show_grid,
                        "b" => self.show_stars = !self.show_stars,
                        "p" => self.show_all = !self.show_all,
                        "l" => self.log = !self.log,
                        "s" => self.true_scale = !self.true_scale,
                        "t" => self.clock.reset_to_now(),
                        // Engine switching. N/G seed the integrator from the
                        // ephemeris only on entry, so N↔G keeps the same state and
                        // trails for a direct comparison.
                        "e" => {
                            self.engine = Engine::Ephemeris;
                            self.state = None;
                        }
                        "n" => {
                            if self.state.is_none() {
                                self.state = Some(seed_state(self.clock.jd()));
                            }
                            self.engine = Engine::Newtonian;
                        }
                        "g" => {
                            if self.state.is_none() {
                                self.state = Some(seed_state(self.clock.jd()));
                            }
                            self.engine = Engine::Relativistic;
                        }
                        // GR strength ×10 / ÷10, to exaggerate the precession.
                        "]" => self.gr_strength = (self.gr_strength * 10.0).min(1.0e9),
                        "[" => self.gr_strength = (self.gr_strength * 0.1).max(1.0e-3),
                        "r" => self.trails.clear(),
                        "k" => {
                            if self.edu.is_some() {
                                self.edu = None;
                            } else {
                                self.edu = Some(edu::Edu::default());
                                // Frame the two-body demo: centre on the Sun.
                                self.camera.target = DVec3::ZERO;
                                self.camera.radius = 3.0;
                                self.camera.theta = 0.6;
                                self.camera.phi = 0.5;
                            }
                        }
                        "y" => self.show_energy = !self.show_energy,
                        "j" => self.show_kepler = !self.show_kepler,
                        "q" => event_loop.exit(),
                        "h" => self.show_manual = !self.show_manual,
                        "?" | "/" => self.show_help = !self.show_help,
                        // Time speed up / down by ×10, clamped to a sensible range.
                        "." => self.change_speed(10.0),
                        "," => self.change_speed(0.1),
                        _ => {}
                    },
                    Key::Named(NamedKey::Space) => self.paused = !self.paused,
                    Key::Named(NamedKey::Tab) => self.cycle_focus(true),
                    Key::Named(NamedKey::F1) => self.show_manual = !self.show_manual,
                    Key::Named(NamedKey::F12) => self.screenshot_requested = true,
                    _ => {}
                }
            }

            _ => {}
        }
    }

    /// Ask for another frame as soon as the event queue is empty.
    ///
    /// What: keeps the animation running by continuously requesting redraws.
    /// How: when winit has finished delivering pending events, we call
    /// `request_redraw`, which schedules another `RedrawRequested` event.
    /// Principle: this turns an event-driven loop into a steady render loop, which
    /// we need because the simulation animates over time.
    /// Units: none.
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

/// Program entry point: create the event loop and run the app.
///
/// What: starts the winit event loop, which drives the whole program.
/// How: build an `EventLoop`, set it to poll continuously (so we animate), then
/// hand control to `run_app`, which calls our [`App`] callbacks until it exits.
/// Principle: a windowed GPU app is structured around the OS event loop; `main`
/// just sets it up and lets it run.
/// Units: none.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    // `Poll` keeps redrawing even with no input, so motion stays smooth.
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(config::load());
    event_loop.run_app(&mut app)?;
    Ok(())
}
