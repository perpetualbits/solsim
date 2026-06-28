# Solar System Simulator

A 3-D solar-system simulator written in Rust, rendering with **wgpu** on
**Wayland**. It shows the real positions of the Sun, the eight planets and the
major moons, with a bright-star background, multiple viewpoints, a time machine,
and a switchable physics engine that demonstrates Mercury's relativistic
perihelion precession. It is aimed at Dutch **4 VWO** level (≈ age 15–16):
clarity and correctness over cleverness.

Built phase by phase; every maths function is documented and unit-tested against
known values.

## Features

- **Real ephemerides** — planets via VSOP87, the Moon via ELP; the true sky for
  any date.
- **All bodies** — 8 planets and 13 major moons (toggle the full system with `P`).
- **Three viewpoints** — top-down (Ecliptic-North), free orbit around any focus
  body, and the sky from a place on Earth (sidereal-time correct; default
  Zutphen).
- **Bright-star background** — the 1500 brightest stars (Yale BSC5), coloured by
  temperature and sized by brightness.
- **Time control** — speed up/slow down, pause, reset; fast inner planets still
  trace smooth orbits (sub-stepped sampling).
- **Physics engine** — analytic ephemeris, or numerically integrated Newtonian
  gravity, or General Relativity (1PN term). With GR you can watch Mercury's orbit
  precess into a rosette.
- **Logarithmic mode** — squash distances so the whole system fits on screen.
- **Adaptive 3-D grid**, fading trails, an in-app manual, and PNG screenshots.

## Build and run (Wayland)

Requirements: a recent Rust toolchain and a Vulkan-capable GPU/driver. The window
is native Wayland (it falls back to X11).

```sh
cargo run --release
```

Or install the binary and run it from anywhere:

```sh
cargo install --path .
solarsim
```

## Controls

| Input | Action |
|---|---|
| Mouse drag | Orbit the camera (Free / Ecliptic-North views) |
| Mouse wheel | Zoom |
| Focus box / `Tab` | Choose / cycle the body the Free view centres on |
| `V` | Cycle viewpoint: Ecliptic-North → Free → Earth-surface |
| `.` / `,` | Time speed ×10 / ÷10 |
| `Space` | Pause / resume |
| `T` | Reset time to now |
| `P` | All planets & moons ↔ just Sun–Earth–Moon |
| `E` / `N` / `G` | Engine: Ephemeris / Newtonian / General Relativity |
| `[` / `]` | GR strength ÷10 / ×10 (exaggerate the precession) |
| `L` | Toggle logarithmic distance mode |
| `C` | Toggle the 3-D grid |
| `B` | Toggle the star background |
| `R` | Clear trails |
| `F1` / `H` | Open the manual (searchable) |
| `F12` | Save a screenshot (`solarsim-NNNN.png`) |
| `?` | Toggle the controls cheat-sheet |

## See Mercury precess

Press `P`, switch to Ecliptic-North (`V`), focus the inner system, then press `G`
(General Relativity) and tap `]` a few times to exaggerate the effect. Speed up
time with `.` — Mercury's trail draws a precessing rosette. Switch to `N`
(Newtonian, same trail) and it closes into a fixed ellipse.

## Configuration

Optional. Put a file `solarsim.conf` next to where you run the program; any line
overrides a default (`#` starts a comment):

```conf
observer_lat = 52.14
observer_lon = 6.20
speed_days_per_sec = 4.63
trail_length = 4000
window_width = 1280
window_height = 800
start_date = 2026-06-29
```

## How the physics works

All astronomy/physics runs in `f64` (AU, days, radians), converted to `f32` only
at the GPU boundary using a camera-centred floating origin. The full reference —
Julian dates, VSOP87, the ecliptic frame, Kepler's equation, Newtonian N-body,
RK4, the 1PN GR term with the 43″/century derivation, star colours, the cameras
and the logarithmic transform — is in [`docs/MATHS.md`](docs/MATHS.md), with each
topic cross-referenced to its source file. The in-app manual (`F1`) is the short
version.

## Tests

```sh
cargo test
```

covers the maths against known values: JD of J2000, Earth ≈ 1 AU, the planet
distances, Kepler's equation, B−V → temperature, and Mercury's GR perihelion
advance (≈ 43″/century, both closed-form and by integration).
