# The Maths & Physics of the Solar System Simulator

This is the complete reference for every formula, algorithm and physical principle
the program uses, with pointers to the source file where each lives. It is written
for a Dutch **4 VWO** reader (≈ age 15–16): plain Unicode maths, every symbol
explained. The in-app manual (press **F1**) is the short version of this document.

## Units and constants

Everything astronomical is computed in **f64** (double precision); values are only
cast to **f32** at the GPU boundary, using a camera-centred *floating origin* so
tiny (Moon-scale) detail survives.

| Quantity | Symbol | Value | Where |
|---|---|---|---|
| Length | AU | astronomical unit (mean Earth–Sun distance) | — |
| Time | day | — | — |
| Angle | rad | radians | — |
| Gaussian gravitational constant | k | 0.017202098950 | `astro/constants.rs` |
| Sun's gravitational parameter | GM_sun = k² | 2.9591×10⁻⁴ AU³/day² | `astro/constants.rs` |
| Speed of light | c | 173.144 AU/day | `astro/constants.rs` |
| Obliquity of the ecliptic | ε | 23.4393° | `astro/constants.rs` |
| 1 AU in km | — | 149 597 870.7 | `astro/constants.rs` |

Each body's gravitational parameter is `G·m = GM_sun · (m / M_sun)`
(`bodies.rs`, `PLANET_GM`).

## 1. Time: Julian Dates — `astro/time.rs`

A **Julian Date (JD)** is the number of days since a fixed epoch. The reference
**J2000** = 2000-01-01 12:00 is `JD = 2451545.0`.

- Calendar → JD uses Meeus' algorithm (via the `astro` crate). We build the
  decimal day ourselves as `day + (hr + min/60 + sec/3600)/24` (the crate's own
  helper mishandles minutes/seconds), then call `julian_day`.
- The system clock (seconds since 1970-01-01) becomes a JD with
  `JD = 2440587.5 + seconds/86400`.
- The clock advances by `Δt_days = realtime_seconds × speed_factor / 86400`.

**Check:** `jd_from_calendar(2000,1,1,12,0,0) = 2451545.0` (unit test).

## 2. Ephemerides — `astro/ephemeris.rs`

An **ephemeris** gives a body's position for any date, with no step-by-step
simulation.

- **Planets — VSOP87.** The `vsop87a` series sum many periodic terms of the form
  `A·cos(B + C·T)`, with `T = (JD − 2451545)/365250` (Julian millennia), giving
  heliocentric **ecliptic rectangular** coordinates (AU) in the fixed J2000 frame.
  Each term is one small periodic shift caused by the planets pulling on each
  other. `planet_position(planet, jd)` selects the right series.
- **Moon — ELP.** `astro`'s lunar theory gives the Moon's geocentric ecliptic
  longitude λ, latitude β and distance r (km). We convert to rectangular
  `(r·cosβ·cosλ, r·cosβ·sinλ, r·sinβ)`, km → AU, and add the Earth's position.
- **Velocity by finite difference.** `velocity_fd(f, jd, δ) = (f(jd+δ) − f(jd−δ))
  / (2δ)` — a central difference, used to seed the physics engine.

**Checks:** Earth–Sun ≈ 1 AU; Earth–Moon ≈ 0.0024–0.0027 AU (unit tests).

## 3. Coordinate frames — `astro/constants.rs`, `render/viewpoints.rs`, `stars/project.rs`

We work in the **ecliptic J2000** frame (Earth's orbital plane, Sun at origin).
Star catalogues use the **equatorial** frame, which differs only by Earth's axial
tilt ε. A single rotation about the shared x-axis converts equatorial → ecliptic:

```
x' = x
y' = y·cosε + z·sinε
z' = −y·sinε + z·cosε
```

The equatorial unit vector from right ascension α and declination δ is
`(cosδ·cosα, cosδ·sinα, sinδ)` (`stars/project.rs`).

## 4. Kepler orbits for the moons — `astro/elements.rs`

Most moons use mean **Keplerian elements** `(a, e, i, Ω, ω, M₀, period)`:

1. Mean anomaly grows linearly: `M = M₀ + 2π·(JD − epoch)/period`.
2. Solve **Kepler's equation** `M = E − e·sin E` for the eccentric anomaly `E` by
   Newton–Raphson: `E ← E − (E − e·sinE − M)/(1 − e·cosE)`.
3. True anomaly `ν = 2·atan2(√(1+e)·sin(E/2), √(1−e)·cos(E/2))`, radius
   `r = a·(1 − e·cosE)`; in-plane point `(r·cosν, r·sinν, 0)`.
4. Rotate by ω (argument of periapsis), i (inclination) and Ω (node) into the
   ecliptic frame, then add the parent planet's position.

This is **Kepler's first law** (an ellipse with the planet at one focus) and
**second law** (faster near periapsis). Positions are approximate mean values.

**Checks:** the solved `E` satisfies Kepler's equation; a circular orbit keeps a
constant radius (unit tests).

## 5. The physics engine — `physics/forces.rs`, `physics/nbody.rs`

When the engine is Newtonian or Relativistic, positions are **computed** by
integrating the equations of motion (the Sun is held fixed at the origin; the
eight planets are integrated; moons ride analytically on their integrated parent).

### Newtonian gravity

Acceleration of body *i* (`forces::accelerations`):

```
a_i = −GM_sun·r_i/|r_i|³  +  Σ_{j≠i} G·m_j·(r_j − r_i)/|r_j − r_i|³
```

The first term is the Sun; the sum is the other planets. This is **Newton's law of
gravitation**, `a = G·m / r²`, written as a vector.

### General-Relativity correction (1PN, Sun-dominated)

Add to each body (r, v relative to the Sun, μ = GM_sun):

```
a_GR = gr_strength · (μ / (c²·r³)) · [ (4μ/r − v²)·r⃗ + 4·(r⃗·v⃗)·v⃗ ]
```

This tiny extra pull makes the orbit slowly rotate (precess). The closed-form
advance per orbit is

```
Δϖ = 6π·μ / (c²·a·(1 − e²))
```

For **Mercury** (a = 0.387099 AU, e = 0.205630, period 87.969 d) this is
**≈ 43″ per century** — the classic test of General Relativity
(`forces::perihelion_advance_arcsec_per_century`). `gr_strength` (keys `[` `]`)
exaggerates it for a quick visual; keep it at 1 for the honest rate.

**Checks:** the closed form gives 43″/century; integrating one Mercury orbit, the
GR term advances the perihelion by the predicted amount while Newtonian keeps a
closed ellipse (unit tests in `physics/nbody.rs`).

### RK4 integrator — `physics/nbody.rs`

The state is `y = (positions, velocities)`; its rate of change is
`(velocities, accelerations)`. **4th-order Runge–Kutta** samples that rate four
times per step (start, two midpoints, end) and combines them 1-2-2-1:

```
y_{n+1} = y_n + (dt/6)·(k1 + 2k2 + 2k3 + k4)
```

This cancels low-order error so orbits stay accurate with large steps. Each frame's
time step is **subdivided** (≈ 0.5-day sub-steps) for accuracy; the analytic engine
is subdivided too, so fast inner planets trace smooth orbits instead of jagged
chords (temporal aliasing). See `main.rs` (`PHYSICS_STEP_DAYS`, `TRAIL_STEP_DAYS`).

## 6. Stars — `stars/color.rs`, `stars/project.rs`

- **Colour index → temperature** (Ballesteros 2012):
  `T = 4600·(1/(0.92·(B−V)+1.7) + 1/(0.92·(B−V)+0.62))` K. Blue stars (B−V ≈ 0)
  ≈ 10000 K; red stars (B−V ≈ 1.5) ≈ 3800 K.
- **Temperature → RGB**: Tanner Helland's black-body fit (hot → blue-white,
  cool → orange-red).
- **Magnitude → size**: magnitude is logarithmic and inverted (smaller = brighter),
  `m₁ − m₂ = −2.5·log₁₀(F₁/F₂)`, so the drawn dot size is
  `size = base·10^(0.2·(m_ref − m))`, clamped.
- **Placement**: (α, δ) → equatorial unit vector → rotate by ε → ecliptic
  direction, painted on a far background sphere that follows camera *rotation* but
  not *position* (`render/starfield.rs`).

The catalogue is the 1500 brightest Yale Bright Star Catalogue stars
(`assets/bsc5.csv`, loaded by `stars/catalog.rs`).

## 7. Cameras and viewpoints — `render/camera.rs`, `render/viewpoints.rs`

- **Orbit camera.** Position = `target + r·(cosφ·cosθ, cosφ·sinθ, sinφ)`; a
  right-handed look-at toward the target with a perspective projection. Near/far
  clip planes scale with the zoom so you can zoom from a moon's surface out to the
  whole system. All built in f64, cast to f32 with the target as a floating origin.
- **Ecliptic-North**: an orbit camera centred on the Sun, starting nearly straight
  down.
- **Earth-surface**: the observer's local **up/north/east** come from **Local
  Sidereal Time** `LST = GMST(jd) + longitude_east`; the zenith points to RA = LST,
  Dec = latitude in the equatorial frame, rotated into the ecliptic. As time runs,
  LST changes and the sky turns. Default observer: Zutphen (52.14°N, 6.20°E).

## 8. The reference grid — `render/grid.rs`

A 3-D lattice of cubes around the focus. Each line **fades with distance** and
stops past a cut-off. The spacing adapts to the zoom in octave steps: the level
nearest `view_scale / IDEAL_CELLS` is brightest, with a twice-finer and a
twice-coarser (thicker) level cross-fading by `frac(log₂ zoom)`. Lines are drawn as
screen-space rectangles (GPU lines are 1 px) so thickness can vary. It is a ruler
only — it never affects the physics.

## 9. Logarithmic mode — `render/logscale.rs`

A **display-only** warp. Each body keeps its direction from the Sun, but its
distance `r` is replaced by

```
r_disp = R₀·ln(1 + r/r₀)
```

Because `ln` grows ever more slowly, far-apart outer planets are pulled into view
without the inner ones collapsing onto the Sun. Applied to bodies and trails at
draw time; the stored positions and the physics are never changed. (Sizes are
boosted and clamped so bodies stay visible at the compressed scale; the grid is
hidden in this mode.)

---

## 10. True-scale toggle — `bodies::real_radius_au`

A **display-only** size switch (key `S`). Normally bodies are drawn far larger than
life so they are visible. True scale instead draws each body at its real radius:

```
radius_AU = radius_km / 149 597 870.7
```

(149 597 870.7 km = 1 AU.) This makes vivid how tiny the bodies are next to the
distances between them — e.g. the Sun ≈ 0.00465 AU, the Earth ≈ 0.0000426 AU.
Positions and physics are untouched; only the drawn radii (and Saturn's ring radius)
change.

## 11. Educational mode — `edu.rs`

A self-contained **two-body demo** (key `K`) that visualises one integration step.
The Sun sits at the origin; one planet starts at `r = 1.2 AU` with 0.85× the
circular speed `v_circ = √(GM_sun/r)`, giving a clear ellipse. Each step uses
**semi-implicit Euler** (deliberately simpler and bigger than the live RK4):

```
a = −GM_sun·r/|r|³  +  gr_strength·(μ/(c²|r|³))·[(4μ/|r| − v²)·r + 4(r·v)·v]
v ← v + a·Δt
r ← r + v·Δt
```

The step is split into five explanation phases, each revealing more vector arrows:
position `r`, velocity drawn as `v·Δt`, gravity's effect as `a·Δt²`, the GR
correction (exaggerated by `GR_ARROW_EXAG` and capped so it is visible), and the
updated velocity. Velocity-type vectors are scaled by `Δt` and acceleration-type by
`Δt²` so all arrows are comparable as *displacement per step* in AU. Turning GR on
makes the orbit slowly precess. This is purely a teaching view; the live simulation
keeps ticking in the background but is not shown.

---

### Source-file map

| Topic | File |
|---|---|
| Constants | `astro/constants.rs` |
| Julian dates, clock | `astro/time.rs` |
| VSOP87 planets, ELP Moon, velocity | `astro/ephemeris.rs` |
| Kepler elements & equation | `astro/elements.rs` |
| Body catalogue & assembly | `bodies.rs` |
| Newtonian + GR forces | `physics/forces.rs` |
| RK4 integrator, state | `physics/nbody.rs` |
| Star colours & sizes | `stars/color.rs` |
| Star placement | `stars/project.rs`, `render/starfield.rs` |
| Orbit camera | `render/camera.rs` |
| Viewpoints, sidereal time | `render/viewpoints.rs` |
| Reference grid | `render/grid.rs` |
| Logarithmic transform | `render/logscale.rs` |
| True-scale radii | `bodies.rs` (`real_radius_au`) |
| Educational mode & vector arrows | `edu.rs`, `render/arrows.rs` |
| Trails | `render/trails.rs` |
| In-app manual | `ui/manual.rs` |
