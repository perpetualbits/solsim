# Solar System Simulator — Design & Claude Code Build Plan

A Rust + Wayland 3D solar-system simulator with real ephemerides, star catalog,
multiple viewpoints, time control, a Newtonian/General-Relativity physics engine,
logarithmic scaling, and built-in + markdown manuals aimed at Dutch **4 VWO** level.

This document is the single source you drive the whole project from. It has four parts:

1. **Architecture & key decisions** — read once.
2. **Math & physics reference** — the formulas Claude Code must implement correctly.
3. **`CLAUDE.md`** — house rules to drop in the repo so every phase obeys them.
4. **Phase prompts** — paste these into Claude Code one at a time, in order.

> **Language note:** comments and manuals default to **English**. To target Dutch
> 4 VWO instead, add one line to the Phase 1 prompt and to `CLAUDE.md`:
> *"Write all doc-comments and manual text in Dutch (Nederlands), 4 VWO level."*

---

## Part 1 — Architecture & Key Decisions

### 1.1 Technology stack (verified current, mid-2026)

| Concern                | Crate(s)                                   | Notes |
|------------------------|--------------------------------------------|-------|
| Window / input (Wayland)| `winit` (0.30+, `ApplicationHandler` API) | Native Wayland; falls back to X11. |
| GPU rendering          | `wgpu` (29.x)                              | Vulkan backend on Wayland. **Breaking release every ~3 months — pin the version.** |
| GUI overlay / manual   | `egui`, `egui-wgpu`, `egui-winit`          | Enable the `wayland` feature. |
| Linear algebra         | `glam`                                     | Use **`DVec3`/`DMat4` (f64)** for physics, `Vec3`/`Mat4` (f32) only at the GPU boundary. |
| Planet ephemerides     | `vsop87`                                   | `vsop87a::*` → heliocentric **rectangular, J2000** (cleanest fixed frame). |
| Moon, time, frames     | `astro`                                    | Julian dates, ELP-2000/82 Moon, equatorial↔ecliptic, sidereal time (Meeus). |
| Vertex byte-casting    | `bytemuck`                                 | `#[repr(C)] Pod` vertex structs. |
| Async init blocking    | `pollster`                                 | `pollster::block_on` for wgpu device request. |

> **Version-pinning gotcha (important):** `egui-wgpu` depends on a *specific* `wgpu`
> version. Add `egui-wgpu` first, then use **the exact `wgpu` version it pulls in**
> for your own renderer. Mismatched `wgpu` versions produce confusing trait errors.

### 1.2 Units & precision

- **Length:** astronomical units (AU). **Time:** days. **Angles:** radians.
- **Gravity constant** in these units: use the Gaussian constant
  `k = 0.01720209895`, so `G·M_sun = k² = 2.959122e-4 AU³/day²`.
  Each body's `G·m = k² · (m / M_sun)`.
- **Speed of light:** `c = 173.144 AU/day` (needed for the GR term).
- **All astronomy/physics in `f64`.** The Sun→Neptune span is ~30 AU but the Moon
  is ~0.0026 AU from Earth; f32 cannot hold both. Convert to f32 **after** the
  camera transform (a "floating origin" centred on the camera target).

### 1.3 The three propagation engines (core design idea)

A single `Engine` enum drives body positions each frame:

- **`Ephemeris`** (default): position = VSOP87(planet) / ELP(Moon) evaluated at the
  current simulated Julian Date. The *real* sky for the *real* date. No integration.
- **`Newtonian`**: on switch, seed each body's position **and velocity** from the
  ephemeris (velocity via finite difference of two ephemeris evaluations a small
  `δt` apart), then integrate with RK4 using Newtonian gravity only.
- **`Relativistic`**: identical integrator **plus** the 1PN Schwarzschild term.

Switching engine **does not clear trails** → you directly compare Newton vs GR and
*see* Mercury's perihelion advance accumulate in the relativistic trail.

A `gr_strength` multiplier (default `1.0`, shown on-screen) scales the GR term so
the precession can be exaggerated for a quick visual; keep `1.0` for the honest rate.

### 1.4 Logarithmic mode (render-only transform)

Never alters physics. At draw time, replace each body's radius-from-Sun `r` with
`r_disp = R0 · ln(1 + r / r0)` (keep the direction unit-vector unchanged), and
compress drawn body **sizes** similarly. Toggled per-frame; trails store *true*
positions and are re-projected through the same transform so they stay consistent.

### 1.5 Module / file layout

```
src/
  main.rs            // winit ApplicationHandler, event loop, wires everything
  app.rs             // App state: clock, engine, camera, toggles, key handling
  astro/
    time.rs          // calendar date <-> Julian Date, SimClock, speed-up
    ephemeris.rs     // VSOP87 planets, ELP Moon, Sun; returns f64 ecliptic-J2000 AU
    frames.rs        // equatorial<->ecliptic, obliquity, local sidereal time
    elements.rs      // Keplerian elements -> position (for outer-planet moons)
    constants.rs     // GM table, radii, colours, k, c, AU, obliquity
  physics/
    nbody.rs         // State vector, RK4 integrator
    forces.rs        // Newtonian accel + 1PN GR correction
  bodies.rs          // Body catalog (name, GM, radius, colour, parent, kind)
  stars/
    catalog.rs       // load BSC5, parse RA/Dec/Vmag/B-V
    color.rs         // B-V -> temperature -> RGB
    project.rs       // (RA,Dec) -> ecliptic unit vector on background sphere
  render/
    gpu.rs           // wgpu init: instance, device, queue, surface, depth buffer
    camera.rs        // orbit camera (theta/phi/zoom) + view/proj matrices
    sphere.rs        // UV-sphere mesh generation, instanced body draw
    trails.rs        // ring-buffer trails, fading line strips
    grid.rs          // faint ecliptic-plane grid
    starfield.rs     // instanced star billboards on the background sphere
    logscale.rs      // the log-distance/size transform helpers
    viewpoints.rs    // Ecliptic-North, Free-orbit, Earth-surface presets
  ui/
    overlay.rs       // egui: HUD (date, speed, engine, gr_strength), controls help
    manual.rs        // in-app manual panels (4 VWO text)
assets/
  bsc5.csv           // trimmed bright-star catalog (1500 rows)
docs/
  MATHS.md           // full markdown manual: every formula, principle, physics
CLAUDE.md            // house rules (Part 3 below)
```

### 1.6 Controls (define once, show in an on-screen help overlay)

| Input                | Action |
|----------------------|--------|
| Mouse drag           | Orbit camera in θ (azimuth) and φ (elevation) |
| Mouse wheel          | Zoom in/out (orbits keep running) |
| `V`                  | Cycle viewpoint: Ecliptic-North → Free → Earth-surface |
| `Space`              | Pause / resume time |
| `.` / `,`            | Time speed ×10 up / down (powers of ten) |
| `T`                  | Reset clock to "now" (system date) |
| `P`                  | Toggle all planets + major moons (vs Sun-Earth-Moon only) |
| `E` / `N` / `G`      | Engine: Ephemeris / Newtonian / General-Relativity |
| `[` / `]`            | GR strength ×0.1 / ×10 |
| `L`                  | Toggle logarithmic distance & size |
| `C`                  | Toggle ecliptic grid |
| `B`                  | Toggle star background |
| `R`                  | Clear trails |
| `F1` or `H`          | Open built-in manual |
| `?`                  | Toggle controls cheat-sheet |

---

## Part 2 — Math & Physics Reference (implement exactly)

These are the formulas the doc-comments must explain and the code must use. Give
each to Claude Code inside the relevant phase prompt; they're collected here so you
can check the output.

**Julian Date.** `JD` of `2000-01-01 12:00 TT` is `2451545.0`. Use the `astro`
crate's `time::julian_day`. Simulation time is a single `f64` JD advanced by
`dt_days = realtime_seconds × speed_factor / 86400`.

**VSOP87 (planets).** Internally sums terms `A·cos(B + C·T)` with
`T = (JD − 2451545)/365250` (Julian millennia). Use `vsop87a` for heliocentric
rectangular J2000 coordinates in AU — a fixed inertial frame, no spherical
conversion needed. The Sun sits at the origin in this frame.

**Moon (ELP-2000/82).** Use `astro`'s lunar module → geocentric Moon position;
add Earth's heliocentric position to place it in the Sun-centred frame.

**Equatorial → ecliptic (for stars).** Build the equatorial unit vector from
right ascension α and declination δ:
`x = cosδ·cosα,  y = cosδ·sinα,  z = sinδ`.
Rotate about the x-axis by the obliquity `ε = 23.4393°` to get ecliptic coords:
```
x' = x
y' = y·cosε + z·sinε
z' = −y·sinε + z·cosε
```
Place stars on a large background sphere (parallax is negligible at solar-system scale).

**B−V colour index → temperature (Ballesteros 2012):**
```
T = 4600 · ( 1/(0.92·(B−V) + 1.7) + 1/(0.92·(B−V) + 0.62) )   [kelvin]
```

**Temperature → RGB.** Use a blackbody approximation (Tanner Helland's piecewise
fit is standard and short). Hot stars → blue-white, cool stars → orange-red.

**Apparent magnitude → disk size.** Magnitude is logarithmic and *inverted*
(smaller = brighter): `m₁ − m₂ = −2.5·log₁₀(F₁/F₂)`. For the drawn radius use
`size = base · 10^(0.2·(m_ref − m))`, clamped to `[size_min, size_max]`
(e.g. `m_ref = 0`). Brighter star → bigger disk.

**Newtonian gravity (the integrator's core force):**
```
a_i = Σ_{j≠i}  G·m_j · (r_j − r_i) / |r_j − r_i|³
```

**General-relativity correction (1PN Schwarzschild, Sun-dominated):** for each
body, with `r` and `v` relative to the Sun, `μ = G·M_sun`, add to its acceleration:
```
a_GR = gr_strength · (μ / (c²·r³)) · [ (4μ/r − v²)·r⃗ + 4·(r⃗·v⃗)·v⃗ ]
```
This reproduces the perihelion advance. Closed-form check (per orbit):
```
Δϖ = 6π·μ / ( c²·a·(1 − e²) )
```
For Mercury this is **43″ per century** — use it as a unit test / sanity number.

**Kepler's equation (for outer-planet moons via mean elements):** solve
`M = E − e·sinE` for eccentric anomaly `E` by Newton–Raphson, then true anomaly
and radius give the position in the orbital plane; rotate by inclination, node,
and argument of periapsis into the ecliptic frame.

**RK4 integrator.** Standard 4th-order Runge–Kutta on the combined state vector
of all bodies (positions + velocities). One substep = `dt_days` clamped so fast
speed-ups subdivide into multiple steps for stability.

**Orbit camera.** Camera position =
`target + r·(cosφ·cosθ, cosφ·sinθ, sinφ)`; build a look-at view matrix toward
`target`, perspective projection, depth buffer enabled.

**Earth-surface viewpoint.** Compute Greenwich Mean Sidereal Time (via `astro`),
add observer longitude → Local Sidereal Time; orient the camera so the observer's
zenith and horizon match the real sky for the given date/time at
**Zutphen (52.14°N, 6.20°E)** by default.

---

## Part 3 — `CLAUDE.md` (paste this into the repo root first)

````markdown
# Project house rules — read before every change

## What this is
A 3D solar-system simulator in Rust with a Wayland GUI. Educational target:
Dutch **4 VWO** (≈ age 15–16, decent maths/physics, not university). Clarity and
correctness beat cleverness.

## Hard rules
- **Edition** Rust 2021. Keep the project compiling and runnable after every phase.
- **Precision:** all astronomy/physics in `f64` (`glam::DVec3`/`DMat4`). Convert to
  `f32` only at the GPU boundary, using a camera-centred floating origin.
- **Units:** AU, days, radians. `k = 0.01720209895`, `GM_sun = k²`,
  `c = 173.144 AU/day`, obliquity `ε = 23.4393°`. Document every constant where defined.
- **Crate versions:** add `egui-wgpu` first; use the exact `wgpu` version it depends
  on for the custom renderer. Pin all versions in `Cargo.toml` (no `"*"`).
- **No `unwrap()`/`expect()`** outside `main` startup. Handle `Result`s.

## Comment & documentation rule (REQUIRED on every function)
Every function gets a `///` doc-comment block, in **English**, written so a 4 VWO
student understands it. Each block states, in this order:
1. **What** the function computes (one sentence).
2. **How** — the algorithm or formula, written out in plain Unicode maths
   (e.g. `a = G·m / r²`), with a sentence on *why* it works.
3. **The principle/physics** behind it (e.g. "Newton's law of gravitation says…").
4. **Units** of inputs and outputs.
Keep it concrete and short; no jargon left unexplained.

## Tests
Add `#[test]`s for every maths function against a known value:
- JD of 2000-01-01 12:00 TT = 2451545.0
- Earth ≈ 1 AU from Sun
- Mercury GR perihelion advance ≈ 43″/century (closed-form check)
- A known star's B−V → temperature in the right ballpark
Use Meeus "Astronomical Algorithms" example values where available.

## Style
- Small modules per the agreed layout. One responsibility each.
- Prefer pure functions for maths; keep GPU/state separate from physics.
- Update the on-screen controls help and `docs/MATHS.md` whenever behaviour changes.
````

---

## Part 4 — Phase Prompts for Claude Code

Paste these **one at a time**, in order. After each, run the app and confirm the
"Done when" check before moving on. Each phase is designed to compile and run.

> **Tip:** start the Claude Code session with: *"Read `CLAUDE.md` and follow it for
> everything. Work phase by phase; stop after each phase so I can test."* Then feed
> the phase prompts below.

---

### Phase 0 — Scaffold a Wayland window with a wgpu clear-screen and an egui overlay

```
Create a new Rust binary project "solarsim". Add winit (0.30+, ApplicationHandler
API), wgpu, egui, egui-wgpu, egui-winit (with the wayland feature), pollster,
bytemuck, glam. Add egui-wgpu first and pin wgpu to the exact version it requires.

Implement main.rs using winit's ApplicationHandler trait:
- open a resizable window titled "Solar System Simulator"
- initialise wgpu (instance, surface, adapter, device, queue) with pollster
- create a depth texture sized to the window
- each frame: clear the colour target to near-black and draw an egui overlay panel
  showing "FPS: <n>" and "Hello, Wayland".
- handle window resize (reconfigure surface + depth texture) and close.

Confirm it builds and runs on Wayland. Follow CLAUDE.md (doc-comment every function).
Done when: a window opens on Wayland showing the FPS overlay on a dark background.
```

---

### Phase 1 — Time and ephemeris core (no graphics yet)

```
Add the `astro` and `vsop87` crates. Create the astro module:

- astro/time.rs: a SimClock holding a Julian Date (f64) and a speed_factor.
  Functions to convert a civil calendar date/time to JD and back (use the astro
  crate), to advance the clock by real seconds × speed_factor, and to set it to the
  system "now". J2000 = 2451545.0.
- astro/constants.rs: k = 0.01720209895, GM_sun = k², c = 173.144 AU/day,
  obliquity = 23.4393°, AU in km, plus a placeholder body table.
- astro/ephemeris.rs: functions returning f64 ecliptic-J2000 rectangular AU
  positions for the Sun (origin), Earth and the Moon (use vsop87a for Earth;
  use astro's ELP lunar position added to Earth's position for the Moon), at a
  given JD. Also a velocity-by-finite-difference helper (evaluate at JD and JD+δ).

In main (temporarily), print the Sun/Earth/Moon positions and Earth–Sun distance
for today's date. Add tests: JD(2000-01-01 12:00 TT)=2451545.0; Earth–Sun distance
≈ 0.98–1.02 AU; Earth–Moon distance ≈ 0.0024–0.0027 AU.
Done when: tests pass and printed Earth–Sun distance is ~1 AU.
```

---

### Phase 2 — Render the Sun–Earth–Moon in 3D with an orbit camera and trails

```
Build the renderer for three bodies at their ephemeris positions.

- render/camera.rs: an orbit camera with spherical params theta, phi, radius and a
  target point. Mouse drag updates theta/phi (clamp phi away from the poles); mouse
  wheel updates radius (zoom). Produce f64 view/projection, then cast to f32 with a
  floating origin centred on the target so Moon-scale detail survives in f32.
- render/sphere.rs: generate a UV-sphere mesh once; draw bodies as instances with
  per-instance position, scale (use exaggerated but readable radii for now), and
  colour. Simple lighting: Sun is emissive (full bright), Earth/Moon are diffuse-lit
  from the Sun's direction.
- render/trails.rs: a fixed-length ring buffer per body storing recent TRUE f64
  positions; draw as a line strip whose alpha fades toward the oldest point.

Each frame: advance the SimClock, get Sun/Earth/Moon positions from the ephemeris,
update instances and trails, render. Keep the egui HUD showing date and Earth–Sun
distance. Default speed_factor so the Moon visibly orbits Earth within a few seconds.
Done when: Sun, Earth and Moon render in 3D, the Moon orbits the Earth which orbits
the Sun, trails fade behind them, and mouse drag/zoom work while motion continues.
```

---

### Phase 3 — Ecliptic grid and the three viewpoints

```
Add render/grid.rs: a faint line grid on the ecliptic plane (z = 0 in the
ecliptic-J2000 frame), centred on the Sun, with concentric circles at 1,2,5,10,20,30
AU plus radial spokes. Toggle with C.

Add render/viewpoints.rs with three presets, cycled by V:
- Ecliptic-North: camera on the +Z ecliptic axis looking straight down at the Sun.
- Free: the existing mouse-controlled orbit camera.
- Earth-surface: place the camera at Earth and orient it using Local Sidereal Time
  (GMST from the astro crate + observer longitude) for observer Zutphen
  (52.14°N, 6.20°E, configurable), so the sky direction is physically correct;
  show a simple horizon line.
Update the HUD to show the active viewpoint and the controls help (toggle with ?).
Done when: V cycles all three views, C toggles the grid, and the grid lies flat in
the ecliptic plane in the top-down view.
```

---

### Phase 4 — Bright-star background

```
Add the 1500 brightest stars.

- Provide assets/bsc5.csv with columns: name, ra_deg, dec_deg, vmag, bv. Source it
  from the Yale Bright Star Catalogue (BSC5), sorted by vmag ascending, top 1500
  rows. (If you cannot fetch it, generate the loader and a small sample, and leave a
  clear TODO with the download URL and parsing notes for the full file.)
- stars/catalog.rs: load and parse the CSV into a Vec of stars.
- stars/color.rs: B−V → temperature (Ballesteros 2012 formula) → RGB (blackbody
  approximation, e.g. Tanner Helland). Magnitude → disk size via
  size = base·10^(0.2·(m_ref − m)), clamped.
- stars/project.rs: (RA,Dec) → equatorial unit vector → rotate by obliquity → ecliptic
  unit vector; place each star on a large background sphere that ignores camera zoom.
- render/starfield.rs: draw stars as instanced camera-facing billboards (or round
  points) with per-star colour and size; render behind everything (no depth write).
Toggle with B.
Done when: a realistic star background appears, hotter stars look bluer, brighter
stars are larger, and the constellations sit correctly relative to the ecliptic and
rotate properly in the Earth-surface view.
```

---

### Phase 5 — All planets and major moons (toggle with P)

```
Extend to the full solar system.

- bodies.rs: a catalog of the 8 planets with GM, mean radius, colour; and major
  moons (Earth: Moon; Mars: Phobos, Deimos; Jupiter: Io, Europa, Ganymede, Callisto;
  Saturn: Titan, Rhea, Iapetus; Uranus: Titania, Oberon; Neptune: Triton), each with
  parent, GM, radius, colour.
- astro/ephemeris.rs: positions for all 8 planets via vsop87a.
- astro/elements.rs: Keplerian mean orbital elements for the listed moons (relative
  to their parent), solved via Kepler's equation (Newton–Raphson) → position, then
  offset by the parent's position. Document that these are approximate mean elements,
  unlike Earth's Moon which uses ELP.
Pressing P toggles between "Sun–Earth–Moon only" and "everything". Body draw scales
must keep tiny moons visible without overlapping their planet.
Done when: P reveals all planets orbiting at correct relative distances/periods with
their moons, and hides them again.
```

---

### Phase 6 — Time speed-up control and logarithmic mode

```
Time controls: keys "." and "," change speed_factor by ×10 / ÷10 across a sensible
range (e.g. 1 second/second up to ~1e8). Space pauses. T resets to now. Show the
current factor in human terms in the HUD (e.g. "1 day/s", "10 years/s"). When the
per-frame step is large, subdivide it into several smaller steps (matters for the
physics engine in Phase 7).

Logarithmic mode (key L): add render/logscale.rs implementing
r_disp = R0·ln(1 + r/r0) on the radius-from-Sun (direction unchanged) and a matching
compression of drawn body sizes. Apply it at draw time to bodies, trails and grid
circles, so distant planets and size ratios become legible. It must NOT change the
stored physics state. Show "LOG" in the HUD when active.
Done when: speed-up lets you watch Neptune complete an orbit, and L compresses the
whole system into one readable view with outer planets no longer off-screen.
```

---

### Phase 7 — Physics engine: Newtonian and General Relativity

```
Add the numerical engine and the E/N/G switch.

- physics/nbody.rs: a State holding f64 position+velocity for every active body, and
  an RK4 integrator stepping by dt_days (subdivided as in Phase 6).
- physics/forces.rs:
  * Newtonian acceleration: a_i = Σ_j G·m_j·(r_j − r_i)/|r_j − r_i|³.
  * GR correction (added per body, r and v relative to the Sun, μ = GM_sun):
    a_GR = gr_strength·(μ/(c²r³))·[ (4μ/r − v²)·r + 4·(r·v)·v ].
- app.rs: Engine enum {Ephemeris, Newtonian, Relativistic}.
  * E = Ephemeris (analytic, default).
  * N / G = seed State from the ephemeris (position + finite-difference velocity) at
    the moment of switching, then integrate Newtonian / Relativistic.
  * Switching engine must NOT clear trails, so Newtonian vs Relativistic trails can
    be compared directly.
  * gr_strength (keys [ and ]) scales the GR term; default 1.0; show it in the HUD.
Add a test: closed-form Mercury advance 6π·μ/(c²·a·(1−e²)) ≈ 43″/century.
Done when: in Relativistic mode with high speed-up (and/or raised gr_strength),
Mercury's persistent trail draws a precessing rosette, while Newtonian mode keeps a
closed ellipse.
```

---

### Phase 8 — Built-in manual and the markdown maths manual

```
Two manuals, both at 4 VWO level.

- ui/manual.rs: an in-app manual (egui window, opened with F1/H) with sections:
  What you are seeing; The viewpoints and controls; What ephemerides are and how
  VSOP87/ELP work (intuitively); Newton's gravity and how the simulation steps time
  (RK4, plainly); What General Relativity adds and why Mercury precesses; The star
  colours and sizes; Logarithmic mode. Use short paragraphs and the on-screen
  symbols. Searchable/scrollable.
- docs/MATHS.md: the complete written manual covering ALL mathematics, algorithms,
  principles and physics actually used: Julian dates; VSOP87 term structure; the
  ecliptic frame and obliquity; equatorial→ecliptic rotation; B−V→temperature→RGB;
  magnitude→size; Newtonian N-body; RK4; the 1PN GR term with the perihelion-advance
  derivation and the 43″/century check; Kepler's equation for moons; the orbit camera
  and sidereal-time Earth-surface view; the logarithmic transform. Include the
  formulas exactly as implemented and cross-reference the source files.
Then audit the whole codebase: ensure EVERY function has the required 4-part
doc-comment block per CLAUDE.md; fix any that are missing or thin.
Done when: F1 opens a readable in-app manual, docs/MATHS.md covers every formula
used, and no function lacks its doc-comment block.
```

---

### Phase 9 — Polish

```
Finishing touches:
- An on-screen controls cheat-sheet (key ?) listing every binding from CLAUDE.md.
- A small config (struct + optional config file) for observer lat/long, default
  date, default speed, trail length, window size.
- A screenshot key (F12) writing a PNG.
- A README.md: what it is, build/run on Wayland, controls, a screenshot, and a short
  "how the physics works" pointing to docs/MATHS.md.
- A final pass for warnings (cargo clippy) and a smoke test that the app starts,
  switches every viewpoint and engine, and exits cleanly.
Done when: clippy is clean, the cheat-sheet lists all controls, and the README lets
a newcomer build and run it on Wayland.
```

---

## Appendix — Sanity checks to keep handy

- Earth–Sun distance oscillates ~0.983 AU (perihelion, early Jan) to ~1.017 AU
  (aphelion, early Jul).
- Moon completes an orbit in ~27.3 days (sidereal).
- Mercury's perihelion advances ~5.0″/orbit ⇒ ~43″/century from GR alone.
- A0 stars (B−V ≈ 0) ≈ 9500 K (blue-white); M stars (B−V ≈ 1.5) ≈ 3500 K (orange-red).
- Neptune's orbital period ≈ 165 years — a good target for the speed-up test.
