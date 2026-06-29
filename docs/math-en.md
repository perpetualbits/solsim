# The Maths & Physics of the Solar System Simulator

*A Dutch translation is available in [`math-nl.md`](math-nl.md).*

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
The step *size* is bounded to `PHYSICS_STEP_DAYS` (`nbody::plan_substeps`) so that
short-period orbits cannot blow up no matter how fast time is requested; at extreme
speed the clock falls behind (with a HUD note) rather than coarsening the step.

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

## 12. Energy, the Hamiltonian, and the virial theorem — `physics/energy.rs`

This section explains the energy graph (key `Y`) and, more importantly, *why* the
equations the simulator integrates are the right ones. It is the deepest part of the
manual; take it slowly.

### 12.1 The two energies

A moving body carries **kinetic energy**, the energy of motion:

```
T = ½·m·v²          (v = |v⃗| is the speed)
```

Gravity stores **potential energy**. For one planet of mass `m` at distance `r`
from the Sun (mass `M`) it is

```
U = −G·M·m / r
```

`U` is *negative* and grows toward 0 as `r → ∞`: that is just bookkeeping for "you
must *add* energy to drag the planet away against the Sun's pull." For the whole
system we add the Sun's pull on every planet **and** every planet–planet pull:

```
T = Σᵢ ½·mᵢ·vᵢ²
U = −Σᵢ G·M·mᵢ/rᵢ  −  Σ_{i<j} G·mᵢ·mⱼ / r_ij
E = T + U                         (the total energy)
```

In our units mass is measured in **solar masses**, so `M = 1` and `G = k² = GM_sun`.
That makes the code tidy: `G·M·mᵢ = gmᵢ` and `G·mᵢ·mⱼ = gmᵢ·gmⱼ / GM_sun`, which is
exactly what `system_energy` computes. (Units of energy here: M_sun·AU²·day⁻².)

### 12.2 Why the total stays flat: conservation of energy

Differentiate `E` with respect to time and use Newton's law `mᵢ·a⃗ᵢ = F⃗ᵢ` (the force
is `F⃗ᵢ = −∂U/∂r⃗ᵢ`):

```
dE/dt = Σᵢ mᵢ·v⃗ᵢ·a⃗ᵢ + Σᵢ (∂U/∂r⃗ᵢ)·v⃗ᵢ
      = Σᵢ v⃗ᵢ·F⃗ᵢ − Σᵢ v⃗ᵢ·F⃗ᵢ = 0
```

So for pure Newtonian gravity the total energy **cannot change**. That is the white
line on the graph: it should be perfectly flat. In practice it drifts a little,
because RK4 with a finite step is only *approximately* exact, and the **exaggerated
GR term is a velocity-dependent force that is not the gradient of a simple potential**
— so with relativity on, energy is not expected to be conserved at all. Watching the
line is therefore a live check on how trustworthy the integration is.

### 12.3 The Hamiltonian — and getting the motion *from* the energy

Write the energy using **momentum** `p⃗ᵢ = mᵢ·v⃗ᵢ` instead of velocity. The result,
seen as a function of positions `r⃗ᵢ` and momenta `p⃗ᵢ`, is the **Hamiltonian**:

```
H(r⃗, p⃗) = Σᵢ |p⃗ᵢ|² / (2·mᵢ)  +  U(r⃗)        (= T + U = E)
```

The whole of the motion follows from `H` through **Hamilton's equations**:

```
dr⃗ᵢ/dt =  ∂H/∂p⃗ᵢ          ṗ⃗ᵢ = −∂H/∂r⃗ᵢ
```

Let us check that these *give back Newton*. The first one:

```
dr⃗ᵢ/dt = ∂H/∂p⃗ᵢ = p⃗ᵢ/mᵢ = v⃗ᵢ          ✓ (momentum is mass × velocity)
```

The second one, using `∂U/∂r⃗ᵢ` for the Sun term `U = −G·M·mᵢ/rᵢ`:

```
ṗ⃗ᵢ = −∂U/∂r⃗ᵢ = −G·M·mᵢ·r⃗ᵢ / rᵢ³
```

Divide by `mᵢ` (since `p⃗ᵢ = mᵢ·v⃗ᵢ`, `ṗ⃗ᵢ/mᵢ = a⃗ᵢ`):

```
a⃗ᵢ = −G·M·r⃗ᵢ / rᵢ³
```

— exactly the acceleration in `physics/forces.rs`. So the simulator is really just
*integrating Hamilton's equations*: the energy `H` **is** the law of motion, and the
gravity formula we code is what you get by differentiating it. This is the modern,
unifying way to see mechanics; the GR correction is an extra term layered on top.

### 12.4 The virial theorem

For gravity (an inverse-square force, where `U ∝ 1/r`) there is a beautiful relation
between the time-averaged energies of a bound orbit:

```
2·⟨T⟩ + ⟨U⟩ = 0          ⇒  ⟨T⟩ = −½·⟨U⟩,  E = ⟨T⟩ + ⟨U⟩ = ½·⟨U⟩ = −⟨T⟩
```

(⟨·⟩ means "averaged over one orbit".) It comes from looking at the quantity
`G = Σ p⃗ᵢ·r⃗ᵢ`: over a full, repeating orbit its average rate of change is zero, and
working that out turns into `2⟨T⟩ + ⟨U⟩ = 0` precisely because the force goes as
`1/r²`. A quick way to trust it: a **circular** orbit satisfies it at *every* instant,
not just on average. There `v² = G·M/r`, so

```
T = ½·m·v² = ½·G·M·m/r,   U = −G·M·m/r   ⇒   2T + U = 0   ✓
```

That circular case is the unit test in `physics/energy.rs`. The theorem explains the
*shape* of the graph: the total `E` sits below zero, roughly halfway down to the
(negative) potential line, and the kinetic line mirrors it above zero. It also has a
famous astrophysical use — weighing star clusters and galaxies from how fast their
members move — but here it is mainly a sanity check on the numbers.

---

## 13. The Milky Way band — `stars/galaxy.rs`, `stars/project.rs`

Galaxies are far too distant to place in an AU-scale scene (the nearest star is
already ~270 000 AU away, and the galaxy is billions of AU across), so the Milky
Way is drawn the same way as the stars: as **directions on the sky**, not objects
at a true distance.

We scatter ~9000 faint stars whose galactic latitude `b` follows a Gaussian
(σ ≈ 6°), so they hug the **galactic plane**, and tint them slightly warmer toward
the Galactic Centre (the bulge). Because the star background uses additive
blending, the overlapping faint glows add up into the soft band you see edge-on
from inside our own galaxy's disk. A fixed random seed makes the band identical
every run, and it is hidden together with the stars (key `B`).

Placing it needs the **galactic → ecliptic** transform (`galactic_to_ecliptic`).
The galactic frame is tilted ≈63° to the equator; it is defined (IAU 1958, J2000)
by the North Galactic Pole at RA 192.85948°, Dec +27.12825° and the Galactic Centre
at RA 266.40499°, Dec −28.93617°. We build an orthonormal basis from those two
directions, express `(l, b)` in it, then rotate by the obliquity ε into the
ecliptic frame — the same final step the stars use.

**Checks:** the North Galactic Pole and Galactic Centre map to their known sky
positions; the generated band's mean |galactic latitude| stays small (unit tests).

---

## 14. Procedural clouds (fractal noise) — `render/clouds.rs`, `render/sphere.rs`

Clouds look the same at many scales, so they are a perfect fit for **fractional
Brownian motion (fBm)**: add several octaves of smooth value noise, each at twice
the frequency and half the strength of the one before. A handful of octaves gives a
rich, self-similar field — and it is cheap, because we bake it once into a texture
at start-up.

```
fbm(p) = Σᵢ amplitudeᵢ · noise(2ⁱ·p),   amplitudeᵢ = ½ⁱ
```

We sample the noise on the **unit sphere** (each pixel's 3-D direction) so the map
wraps seamlessly and does not pinch at the poles, swirl it with a **domain warp**
(offset the sample point by a second noise field) for wind-sheared streaks, and turn
the fBm value into a soft coverage with `smoothstep`. That coverage is stored as the
texture's alpha.

The Earth then gets a thin translucent **cloud shell**: a sphere ~2% larger than the
planet, drawn after the solid bodies with alpha blending and no depth writing, lit by
the same Sun (so clouds fade at the terminator) and spun a few percent faster than
the surface so the weather drifts. It is hidden only when the Earth itself is (log
mode, or the surface view where you stand on it).

**Checks:** value noise and fBm stay in 0..1; the baked map contains both clear sky
and solid cloud (unit tests).

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
| Energy, Hamiltonian, virial | `physics/energy.rs`, `ui/energy.rs` |
| Star colours & sizes | `stars/color.rs` |
| Star placement | `stars/project.rs`, `render/starfield.rs` |
| Milky Way band, galactic coordinates | `stars/galaxy.rs`, `stars/project.rs` |
| Procedural clouds (fBm) | `render/clouds.rs`, `render/sphere.rs` |
| Orbit camera | `render/camera.rs` |
| Viewpoints, sidereal time | `render/viewpoints.rs` |
| Reference grid | `render/grid.rs` |
| Logarithmic transform | `render/logscale.rs` |
| True-scale radii | `bodies.rs` (`real_radius_au`) |
| Educational mode & vector arrows | `edu.rs`, `render/arrows.rs` |
| Trails | `render/trails.rs` |
| In-app manual | `ui/manual.rs` |
