//! The in-app manual: a searchable, scrollable help window (opened with F1/H).
//!
//! The text is written for a Dutch 4-VWO reader (≈ age 15–16): short paragraphs,
//! plain Unicode maths, every symbol explained. It mirrors the deeper reference in
//! `docs/math-en.md`.

/// The manual's sections, as `(title, body)` pairs shown in order.
///
/// What: the full text of the built-in manual.
/// How/why: keeping it as a simple table lets the window filter by a search box
/// (matching the title or body) and render each section the same way.
/// Units: none (text).
const SECTIONS: &[(&str, &str)] = &[
    (
        "What you are seeing",
        "This is a 3-D model of the solar system at real scale. The Sun sits at the \
centre; the planets orbit it, and moons orbit the planets. Distances are measured \
in astronomical units (AU): 1 AU is the average Earth–Sun distance. Bodies are \
drawn far larger than reality so you can see them — otherwise they would be \
invisible dots. Their positions, however, are real.",
    ),
    (
        "Viewpoints and controls",
        "Press V to cycle three viewpoints:\n\
• Ecliptic-North — looking straight down on the system from above the Sun.\n\
• Free — fly around the body you have focused on (drag to rotate, wheel to zoom).\n\
• Earth-surface — the sky as seen from a place on Earth (Zutphen), which turns as \
time passes.\n\n\
Pick the focus body in the panel, or step through with Tab. Time runs with the keys \
'.' and ',' (×10 faster / slower), Space pauses, and T jumps back to today. P shows \
all planets and moons or just the Sun–Earth–Moon. C, B and L toggle the grid, the \
stars and logarithmic mode. S switches body sizes between 'exaggerated' (easy to \
see) and 'true' (their real radii). K opens the educational mode, and Y the energy \
graph. Q quits the program.",
    ),
    (
        "Time and Julian dates",
        "Astronomers count time as a single number, the Julian Date (JD): days since \
a fixed moment long ago. The reference 'J2000' is 1 January 2000, 12:00, which is \
JD 2451545.0. Using one number makes the maths easy — to move forward you just add \
days. Speeding up time multiplies how many simulated days pass per real second.",
    ),
    (
        "Ephemerides: VSOP87 and ELP",
        "An 'ephemeris' is a recipe that tells you where a body is on any date. For \
the planets we use VSOP87, which adds up many small waves of the form A·cos(B + C·T) \
(T is the time since J2000). Each wave captures one way a planet's orbit slowly \
shifts because the other planets tug on it. For the Moon we use the ELP theory, \
which does the same thing for the Moon's more complicated path around the Earth. \
These give the *real* sky for the *real* date, without any step-by-step simulation.",
    ),
    (
        "Coordinates: the ecliptic and the tilt",
        "We measure positions in the ecliptic frame: the flat plane of the Earth's \
orbit, with the Sun at the origin. Stars come listed in a different frame (the \
equatorial one, tied to the Earth's equator). The two differ only by the Earth's \
axial tilt ε ≈ 23.44° (the obliquity), so one rotation about the x-axis converts \
between them: x' = x, y' = y·cosε + z·sinε, z' = −y·sinε + z·cosε. This tilt is also \
why we have seasons.",
    ),
    (
        "Newton's gravity and stepping time (RK4)",
        "Switch the engine to Newtonian with N. Now the program no longer reads \
positions from a formula — it computes them. Newton's law of gravitation says every \
mass pulls every other with a = G·m / r²: stronger for nearer, heavier bodies. From \
the pull we get the acceleration, which changes the velocity, which changes the \
position. We step this forward in tiny time steps using 'RK4' (4th-order \
Runge–Kutta), which samples the pull four times across each step and averages them \
so the orbits stay accurate.",
    ),
    (
        "General Relativity and Mercury",
        "Press G for the relativistic engine. Einstein's theory adds a small extra \
term to the Sun's pull. Its effect is tiny but it makes an orbit slowly rotate \
(precess) instead of closing perfectly. For Mercury this is 43 arc-seconds per \
century — a famous early proof of relativity. With the keys [ and ] you can \
exaggerate the effect (gr-strength); the orbit's trail then draws a 'rosette' you \
can watch form, while the Newtonian engine keeps a closed ellipse.",
    ),
    (
        "Moons: Kepler's equation",
        "Most moons here follow simple average ('mean') orbits. An orbit is an \
ellipse (Kepler's first law), and a body speeds up when it is closer to its planet \
(Kepler's second law). To find a moon's place at a time we solve Kepler's equation \
M = E − e·sin E for the angle E, using a few rounds of Newton's method, then turn E \
into the real position. These positions are approximate but show the moons circling \
at the right distance and speed.",
    ),
    (
        "Star colours and sizes",
        "A star's colour tells its temperature. From the catalogue's B−V 'colour \
index' we estimate the temperature with the Ballesteros formula, then turn that \
into a colour: hot stars (≈10000 K) look blue-white, cool stars (≈3500 K) \
orange-red. A star's brightness is its 'magnitude', which is backwards — smaller \
means brighter — and logarithmic. We draw brighter stars as bigger dots using \
size = base·10^(0.2·(m_ref − m)).",
    ),
    (
        "The reference grid",
        "Press C for a faint 3-D grid of cubes around what you are looking at. It \
fades with distance so only the nearby cells show, and its spacing adapts to the \
zoom (with a twice-finer and twice-coarser level fading in and out) so it stays \
useful at every scale. It is only a ruler — it has no effect on the physics.",
    ),
    (
        "Logarithmic mode",
        "Press L to squash distances logarithmically: each body keeps its true \
direction from the Sun, but its distance r is replaced by R₀·ln(1 + r/r₀). Because \
the logarithm grows ever more slowly, the far-apart outer planets are pulled into \
view without the inner ones piling onto the Sun — the whole system fits on screen at \
once. This changes only the picture, never the real positions or the physics.",
    ),
    (
        "True scale: how tiny everything really is",
        "Press S to switch from the exaggerated drawing sizes to each body's *real* \
radius (its size in km ÷ 149 597 871, the number of km in one AU). Suddenly the \
planets shrink to specks: the Sun's radius is only about 0.0047 AU, and the Earth \
about 0.000043 AU — far smaller than the gaps between them. This is the honest \
picture of how empty space is. The normal mode exaggerates sizes only so you can \
see the bodies; their positions are always real.",
    ),
    (
        "Energy: is it conserved?",
        "Press Y for a graph of the planets' energy. A moving body has kinetic \
energy KE = ½·m·v² (energy of motion); gravity stores potential energy \
PE = −G·M·m/r (negative, because you must add energy to pull bodies apart). The \
key idea is that their sum, the total energy, cannot change for a closed system — \
so the white 'total' line should stay flat. If it slopes or wobbles, that is the \
simulation drifting: bigger time steps and the exaggerated relativity term both \
break perfect conservation, which is exactly why the graph is worth watching. \
There is also the virial theorem: averaged over an orbit, 2·KE + PE = 0, so a \
bound orbit always has more negative potential than positive kinetic energy. See \
docs/math-en.md for how the equations of motion follow from this energy (the \
Hamiltonian).",
    ),
    (
        "Educational mode: one step at a time",
        "Press K for a guided two-body demo (just the Sun and one planet) that shows \
how the simulation actually works. It walks through one time step in five stages, \
with a big arrow for every vector: where the planet is (yellow, the position r), \
how fast it moves (green, the velocity v·Δt), gravity's pull bending the path (red, \
a·Δt²), Einstein's tiny extra pull (magenta, exaggerated so you can see it), and \
the updated velocity. Use 'Next' to advance a stage, 'Step' to take the step and \
watch the orbit grow, or 'Play' to run it. Turn general relativity on and watch the \
orbit slowly turn (precess). It is the same loop — compute the pull, nudge the \
velocity, move, repeat — that runs the whole simulator, only with big visible steps \
instead of tiny RK4 ones. Press K again to leave.",
    ),
];

/// Draw the manual window if it is open.
///
/// What: shows a scrollable, searchable help window.
/// How/why: a text box filters the sections (matching the title or body, ignoring
/// case); the matching sections are listed inside a scroll area. The `open` flag is
/// driven by egui's window close button so the caller can keep it in sync.
/// Principle: immediate-mode UI — we rebuild the whole window from the current
/// search text every frame.
/// Units: none; `search` is the current filter text.
pub fn show(ctx: &egui::Context, open: &mut bool, search: &mut String) {
    egui::Window::new("Manual")
        .open(open)
        .default_size([540.0, 620.0])
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Search:");
                ui.text_edit_singleline(search);
                if ui.button("clear").clicked() {
                    search.clear();
                }
            });
            ui.separator();

            let query = search.to_lowercase();
            egui::ScrollArea::vertical().show(ui, |ui| {
                let mut shown = 0;
                for (title, body) in SECTIONS {
                    let matches = query.is_empty()
                        || title.to_lowercase().contains(&query)
                        || body.to_lowercase().contains(&query);
                    if !matches {
                        continue;
                    }
                    shown += 1;
                    ui.heading(*title);
                    ui.label(*body);
                    ui.add_space(10.0);
                }
                if shown == 0 {
                    ui.label("No sections match your search.");
                }
            });
        });
}
