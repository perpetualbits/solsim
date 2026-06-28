//! The catalogue of bodies: the Sun, the eight planets and the major moons.
//!
//! Each body knows how to be drawn (colour, exaggerated size, whether it glows)
//! and how to find its position (the Sun is the origin, planets come from VSOP87,
//! the Earth's Moon from the ELP theory, and the other moons from mean Keplerian
//! elements). The list is fixed; the `P` key just chooses whether to show the
//! whole system or only the Sun–Earth–Moon "core".

use glam::DVec3;

use crate::astro::constants::GM_SUN;
use crate::astro::elements::Elements;
use crate::astro::ephemeris::{self, Planet};

/// The Julian Date the moon elements are referenced to (J2000.0).
const EPOCH: f64 = 2_451_545.0;

/// The eight planets, in catalogue order (matching [`BODIES`] indices 1..=8).
///
/// What: the planet identifiers the integrator and ephemeris iterate over.
/// How/why: keeping this list lets the physics engine seed and step the planets
/// without re-deriving which catalogue entries are planets.
/// Units: none.
pub const PLANETS: [Planet; 8] = [
    Planet::Mercury,
    Planet::Venus,
    Planet::Earth,
    Planet::Mars,
    Planet::Jupiter,
    Planet::Saturn,
    Planet::Uranus,
    Planet::Neptune,
];

/// Gravitational parameters `G·m` of the eight planets, in AU³·day⁻².
///
/// What: each planet's mass, expressed as the product the force law needs.
/// How/why: `G·m = GM_sun · (m / M_sun)`, so we scale the Sun's value by each
/// planet's mass ratio (IAU values). These matter only for the small planet-on-
/// planet tugs; the Sun dominates everything.
/// Units: AU³·day⁻², in the same order as [`PLANETS`].
pub const PLANET_GM: [f64; 8] = [
    GM_SUN * 1.6601e-7, // Mercury
    GM_SUN * 2.4478e-6, // Venus
    GM_SUN * 3.0035e-6, // Earth
    GM_SUN * 3.2272e-7, // Mars
    GM_SUN * 9.5479e-4, // Jupiter
    GM_SUN * 2.8588e-4, // Saturn
    GM_SUN * 4.3662e-5, // Uranus
    GM_SUN * 5.1514e-5, // Neptune
];

/// Where a body's position comes from.
///
/// What: the recipe for finding one body in space.
/// How/why: different bodies need different methods — the Sun is fixed at the
/// centre, planets and the Earth's Moon have dedicated theories, and the remaining
/// moons use mean elements relative to their parent planet.
/// Units: none (a tag).
pub enum Source {
    /// The Sun, fixed at the origin.
    Sun,
    /// A planet (filled in by slot from [`PLANETS`] / the integrator).
    Planet,
    /// The Earth's Moon, via the ELP theory (already heliocentric).
    ElpMoon,
    /// A moon orbiting `parent` (an index into [`BODIES`]) by Keplerian elements.
    Satellite { parent: usize, elements: Elements },
}

/// One entry in the body catalogue.
///
/// What: everything needed to draw and place a body.
/// How/why: bundling the look (colour, size, glow) with the position recipe keeps
/// the catalogue in one readable table.
/// Units: `color` linear RGB; `draw_radius_au` in AU (exaggerated for visibility);
/// `emissive` true for self-lit bodies; `core` true for the Sun–Earth–Moon set.
pub struct BodyDef {
    /// Body name; not shown yet, kept for future labels and the manual.
    #[allow(dead_code)]
    pub name: &'static str,
    pub color: [f32; 3],
    pub draw_radius_au: f32,
    pub emissive: bool,
    pub core: bool,
    pub source: Source,
}

/// Index of the Earth in [`BODIES`] (the default camera target).
pub const EARTH_INDEX: usize = 3;

/// The full catalogue: Sun, eight planets, then the major moons.
///
/// Draw radii are exaggerated so bodies are visible, but each planet's radius is
/// kept smaller than its innermost listed moon's orbit so moons do not vanish
/// inside their planet. (Mars's moons orbit so close that they sit on its drawn
/// disc — an unavoidable artefact of fixed real distances.) Moon elements are
/// approximate mean values, unlike the Earth's Moon which uses ELP, and several
/// moon inclinations are given relative to their planet's equator rather than the
/// ecliptic — fine for showing them circle at the right distance and speed.
pub const BODIES: [BodyDef; 22] = [
    BodyDef {
        name: "Sun",
        color: [1.0, 0.85, 0.30],
        draw_radius_au: 0.03,
        emissive: true,
        core: true,
        source: Source::Sun,
    },
    BodyDef {
        name: "Mercury",
        color: [0.62, 0.58, 0.52],
        draw_radius_au: 0.0008,
        emissive: false,
        core: false,
        source: Source::Planet,
    },
    BodyDef {
        name: "Venus",
        color: [0.9, 0.8, 0.5],
        draw_radius_au: 0.0011,
        emissive: false,
        core: false,
        source: Source::Planet,
    },
    BodyDef {
        name: "Earth",
        color: [0.25, 0.5, 1.0],
        draw_radius_au: 0.0009,
        emissive: false,
        core: true,
        source: Source::Planet,
    },
    BodyDef {
        name: "Mars",
        color: [0.8, 0.35, 0.2],
        draw_radius_au: 0.0008,
        emissive: false,
        core: false,
        source: Source::Planet,
    },
    BodyDef {
        name: "Jupiter",
        color: [0.82, 0.71, 0.55],
        draw_radius_au: 0.0018,
        emissive: false,
        core: false,
        source: Source::Planet,
    },
    BodyDef {
        name: "Saturn",
        color: [0.85, 0.78, 0.55],
        draw_radius_au: 0.0016,
        emissive: false,
        core: false,
        source: Source::Planet,
    },
    BodyDef {
        name: "Uranus",
        color: [0.6, 0.85, 0.9],
        draw_radius_au: 0.0012,
        emissive: false,
        core: false,
        source: Source::Planet,
    },
    BodyDef {
        name: "Neptune",
        color: [0.3, 0.45, 0.9],
        draw_radius_au: 0.0012,
        emissive: false,
        core: false,
        source: Source::Planet,
    },
    // --- Moons -------------------------------------------------------------
    BodyDef {
        name: "Moon",
        color: [0.72, 0.72, 0.74],
        draw_radius_au: 0.00035,
        emissive: false,
        core: true,
        source: Source::ElpMoon,
    },
    BodyDef {
        name: "Phobos",
        color: [0.55, 0.5, 0.46],
        draw_radius_au: 0.00025,
        emissive: false,
        core: false,
        source: Source::Satellite {
            parent: 4,
            elements: Elements {
                a: 6.27e-5,
                e: 0.0151,
                inc_deg: 1.08,
                node_deg: 0.0,
                peri_deg: 0.0,
                m0_deg: 20.0,
                period: 0.3189,
                epoch: EPOCH,
            },
        },
    },
    BodyDef {
        name: "Deimos",
        color: [0.55, 0.5, 0.46],
        draw_radius_au: 0.00025,
        emissive: false,
        core: false,
        source: Source::Satellite {
            parent: 4,
            elements: Elements {
                a: 1.568e-4,
                e: 0.00033,
                inc_deg: 1.79,
                node_deg: 0.0,
                peri_deg: 0.0,
                m0_deg: 200.0,
                period: 1.2624,
                epoch: EPOCH,
            },
        },
    },
    BodyDef {
        name: "Io",
        color: [0.9, 0.85, 0.45],
        draw_radius_au: 0.00035,
        emissive: false,
        core: false,
        source: Source::Satellite {
            parent: 5,
            elements: Elements {
                a: 2.819e-3,
                e: 0.0041,
                inc_deg: 0.036,
                node_deg: 0.0,
                peri_deg: 0.0,
                m0_deg: 0.0,
                period: 1.7691,
                epoch: EPOCH,
            },
        },
    },
    BodyDef {
        name: "Europa",
        color: [0.85, 0.8, 0.7],
        draw_radius_au: 0.00035,
        emissive: false,
        core: false,
        source: Source::Satellite {
            parent: 5,
            elements: Elements {
                a: 4.486e-3,
                e: 0.0094,
                inc_deg: 0.466,
                node_deg: 0.0,
                peri_deg: 0.0,
                m0_deg: 90.0,
                period: 3.5512,
                epoch: EPOCH,
            },
        },
    },
    BodyDef {
        name: "Ganymede",
        color: [0.7, 0.68, 0.62],
        draw_radius_au: 0.00045,
        emissive: false,
        core: false,
        source: Source::Satellite {
            parent: 5,
            elements: Elements {
                a: 7.155e-3,
                e: 0.0013,
                inc_deg: 0.177,
                node_deg: 0.0,
                peri_deg: 0.0,
                m0_deg: 180.0,
                period: 7.1546,
                epoch: EPOCH,
            },
        },
    },
    BodyDef {
        name: "Callisto",
        color: [0.55, 0.52, 0.5],
        draw_radius_au: 0.00042,
        emissive: false,
        core: false,
        source: Source::Satellite {
            parent: 5,
            elements: Elements {
                a: 1.2585e-2,
                e: 0.0074,
                inc_deg: 0.192,
                node_deg: 0.0,
                peri_deg: 0.0,
                m0_deg: 270.0,
                period: 16.689,
                epoch: EPOCH,
            },
        },
    },
    BodyDef {
        name: "Titan",
        color: [0.85, 0.7, 0.4],
        draw_radius_au: 0.00042,
        emissive: false,
        core: false,
        source: Source::Satellite {
            parent: 6,
            elements: Elements {
                a: 8.168e-3,
                e: 0.0288,
                inc_deg: 0.349,
                node_deg: 0.0,
                peri_deg: 0.0,
                m0_deg: 30.0,
                period: 15.945,
                epoch: EPOCH,
            },
        },
    },
    BodyDef {
        name: "Rhea",
        color: [0.75, 0.74, 0.72],
        draw_radius_au: 0.0003,
        emissive: false,
        core: false,
        source: Source::Satellite {
            parent: 6,
            elements: Elements {
                a: 3.523e-3,
                e: 0.001,
                inc_deg: 0.345,
                node_deg: 0.0,
                peri_deg: 0.0,
                m0_deg: 150.0,
                period: 4.5182,
                epoch: EPOCH,
            },
        },
    },
    BodyDef {
        name: "Iapetus",
        color: [0.6, 0.55, 0.5],
        draw_radius_au: 0.0003,
        emissive: false,
        core: false,
        source: Source::Satellite {
            parent: 6,
            elements: Elements {
                a: 2.381e-2,
                e: 0.0286,
                inc_deg: 15.47,
                node_deg: 0.0,
                peri_deg: 0.0,
                m0_deg: 250.0,
                period: 79.33,
                epoch: EPOCH,
            },
        },
    },
    BodyDef {
        name: "Titania",
        color: [0.7, 0.72, 0.72],
        draw_radius_au: 0.0003,
        emissive: false,
        core: false,
        source: Source::Satellite {
            parent: 7,
            elements: Elements {
                a: 2.913e-3,
                e: 0.0011,
                inc_deg: 0.34,
                node_deg: 0.0,
                peri_deg: 0.0,
                m0_deg: 60.0,
                period: 8.7062,
                epoch: EPOCH,
            },
        },
    },
    BodyDef {
        name: "Oberon",
        color: [0.66, 0.66, 0.66],
        draw_radius_au: 0.0003,
        emissive: false,
        core: false,
        source: Source::Satellite {
            parent: 7,
            elements: Elements {
                a: 3.904e-3,
                e: 0.0014,
                inc_deg: 0.058,
                node_deg: 0.0,
                peri_deg: 0.0,
                m0_deg: 240.0,
                period: 13.463,
                epoch: EPOCH,
            },
        },
    },
    BodyDef {
        name: "Triton",
        color: [0.7, 0.78, 0.8],
        draw_radius_au: 0.00035,
        emissive: false,
        core: false,
        source: Source::Satellite {
            parent: 8,
            elements: Elements {
                a: 2.371e-3,
                e: 0.000016,
                inc_deg: 156.9, // retrograde, steeply inclined (approximate)
                node_deg: 0.0,
                peri_deg: 0.0,
                m0_deg: 0.0,
                period: 5.877,
                epoch: EPOCH,
            },
        },
    },
];

/// The eight planets' heliocentric positions from the analytic ephemeris.
///
/// What: VSOP87 positions for Mercury…Neptune, in [`PLANETS`] order.
/// How/why: used directly in ephemeris mode, and to seed the integrator.
/// Units: `jd` in days; returns positions in AU.
pub fn planet_positions(jd: f64) -> [DVec3; 8] {
    let mut p = [DVec3::ZERO; 8];
    for (slot, planet) in PLANETS.iter().enumerate() {
        p[slot] = ephemeris::planet_position(*planet, jd);
    }
    p
}

/// Assemble the full body list from a set of planet positions.
///
/// What: returns a position for every entry of [`BODIES`], given where the planets
/// are.
/// How/why: the Sun is the origin; the planets are filled in from `planet_pos`
/// (whether those came from the ephemeris or the integrator); then each moon is
/// placed relative to its (now-known) parent — the Earth's Moon by its geocentric
/// ELP offset, the others by their Keplerian offset. Sharing this between the
/// ephemeris and the physics engine keeps the moons consistent in both.
/// Units: `jd` in days; `planet_pos` in AU (order of [`PLANETS`]); returns
/// positions in AU.
pub fn assemble(jd: f64, planet_pos: &[DVec3]) -> Vec<DVec3> {
    let mut pos = vec![DVec3::ZERO; BODIES.len()];
    let n = PLANETS.len().min(planet_pos.len()); // BODIES index 1..=8 are the planets
    pos[1..1 + n].copy_from_slice(&planet_pos[..n]);
    for (i, body) in BODIES.iter().enumerate() {
        match &body.source {
            // Geocentric Moon (ELP) added to wherever the Earth currently is.
            Source::ElpMoon => {
                let geocentric = ephemeris::moon_position(jd) - ephemeris::earth_position(jd);
                pos[i] = pos[EARTH_INDEX] + geocentric;
            }
            Source::Satellite { parent, elements } => {
                pos[i] = pos[*parent] + elements.position(jd);
            }
            _ => {}
        }
    }
    pos
}

/// Compute every body's position from the analytic ephemeris.
///
/// What: convenience wrapper — planet positions from VSOP87, then [`assemble`].
/// Units: `jd` in days; returns positions in AU (ecliptic-J2000, Sun-centred).
pub fn system_positions(jd: f64) -> Vec<DVec3> {
    assemble(jd, &planet_positions(jd))
}

#[cfg(test)]
mod tests {
    use super::*;

    const J2000: f64 = 2_451_545.0;

    /// Each planet should sit within its real range of distances from the Sun.
    #[test]
    fn planet_distances_reasonable() {
        let p = system_positions(J2000);
        let d = |i: usize| p[i].length();
        assert!((0.30..0.47).contains(&d(1)), "Mercury {}", d(1));
        assert!((0.71..0.73).contains(&d(2)), "Venus {}", d(2));
        assert!((0.98..1.02).contains(&d(3)), "Earth {}", d(3));
        assert!((1.36..1.68).contains(&d(4)), "Mars {}", d(4));
        assert!((4.9..5.5).contains(&d(5)), "Jupiter {}", d(5));
        assert!((9.0..10.2).contains(&d(6)), "Saturn {}", d(6));
        assert!((18.2..20.2).contains(&d(7)), "Uranus {}", d(7));
        assert!((29.6..30.5).contains(&d(8)), "Neptune {}", d(8));
    }

    /// Moons should orbit their parent at about their semi-major-axis distance.
    #[test]
    fn moons_orbit_their_parent() {
        let p = system_positions(J2000);
        let io = (p[12] - p[5]).length(); // Io around Jupiter
        assert!((0.0026..0.0030).contains(&io), "Io–Jupiter {io}");
        let titan = (p[16] - p[6]).length(); // Titan around Saturn
        assert!((0.0078..0.0086).contains(&titan), "Titan–Saturn {titan}");
        let triton = (p[21] - p[8]).length(); // Triton around Neptune
        assert!((0.0022..0.0026).contains(&triton), "Triton–Neptune {triton}");
    }
}
