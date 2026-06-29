//! Orbits from Keplerian elements (used for the outer-planet moons).
//!
//! For most moons we do not have a high-precision theory like the Earth's Moon
//! (ELP), so we use **mean orbital elements**: the average size, shape and tilt of
//! the orbit. From those we work out where the moon is at a given time by solving
//! Kepler's equation. These positions are *approximate* — good enough to see the
//! moons circling their planet at the right distance and speed, but not to the
//! second.

use glam::DVec3;

/// Mean Keplerian elements of an orbit around a parent body.
///
/// What: the six numbers that describe an ellipse in space, plus its period and
/// the moment they are measured.
/// How/why: `a` and `e` set the ellipse's size and shape; `inc`, `node` and `peri`
/// tilt and turn it; `m0` says where the body is along it at `epoch`; `period`
/// says how fast it goes round. Together they pin down the orbit.
/// Units: `a` in AU; `e` dimensionless; `inc_deg`/`node_deg`/`peri_deg`/`m0_deg`
/// in degrees; `period` in days; `epoch` is a Julian Date (days).
#[derive(Clone, Copy)]
pub struct Elements {
    /// Semi-major axis (half the long diameter of the ellipse), AU.
    pub a: f64,
    /// Eccentricity (0 = circle, →1 = very stretched).
    pub e: f64,
    /// Inclination: tilt of the orbit plane, degrees.
    pub inc_deg: f64,
    /// Longitude of the ascending node Ω: where the orbit crosses the reference
    /// plane going north, degrees.
    pub node_deg: f64,
    /// Argument of periapsis ω: angle from the node to the closest point, degrees.
    pub peri_deg: f64,
    /// Mean anomaly at `epoch`: how far along the orbit the body starts, degrees.
    pub m0_deg: f64,
    /// Orbital period, days.
    pub period: f64,
    /// Epoch the elements refer to, as a Julian Date (days).
    pub epoch: f64,
}

/// Solve Kepler's equation `M = E − e·sin E` for the eccentric anomaly `E`.
///
/// What: finds the angle `E` that matches a given mean anomaly `M`.
/// How/why: there is no neat formula, so we use Newton–Raphson — start from a
/// guess and repeatedly improve it with `E ← E − (E − e·sinE − M)/(1 − e·cosE)`,
/// which is the "slide down the tangent line" root-finder. It converges in a
/// handful of steps for the small eccentricities of moons.
/// Principle: Kepler's second law (equal areas in equal times) makes the body
/// speed up near periapsis; `E` is the mathematical bridge from the steadily
/// growing `M` to the real position.
/// Units: `m` and the returned `E` in radians; `e` dimensionless.
pub fn solve_kepler(m: f64, e: f64) -> f64 {
    let m = m.rem_euclid(std::f64::consts::TAU);
    // A good starting guess: M for nearly circular orbits, π for stretched ones.
    let mut ecc_anom = if e < 0.8 { m } else { std::f64::consts::PI };
    for _ in 0..60 {
        let delta = (ecc_anom - e * ecc_anom.sin() - m) / (1.0 - e * ecc_anom.cos());
        ecc_anom -= delta;
        if delta.abs() < 1e-12 {
            break;
        }
    }
    ecc_anom
}

impl Elements {
    /// Position of the orbiting body relative to its parent at a Julian Date.
    ///
    /// What: returns the offset vector from the parent to the moon.
    /// How/why: (1) advance the mean anomaly `M = M₀ + 2π·(JD − epoch)/period`;
    /// (2) solve Kepler's equation for `E`; (3) get the true anomaly `ν` and radius
    /// `r = a·(1 − e·cosE)`, giving the point in the orbit's own plane; (4) rotate
    /// that point by the argument of periapsis ω, the inclination i, and the node Ω
    /// into the shared ecliptic frame. The caller then adds the parent's position.
    /// Principle: Kepler's first law — a moon traces an ellipse with its planet at
    /// one focus.
    /// Units: `jd` in days; returns a position in AU (ecliptic frame, relative to
    /// the parent).
    pub fn position(&self, jd: f64) -> DVec3 {
        let mean =
            self.m0_deg.to_radians() + std::f64::consts::TAU * (jd - self.epoch) / self.period;
        let ecc = solve_kepler(mean, self.e);

        // True anomaly ν and radius r in the orbital plane.
        let half = ecc / 2.0;
        let nu = 2.0
            * f64::atan2(
                (1.0 + self.e).sqrt() * half.sin(),
                (1.0 - self.e).sqrt() * half.cos(),
            );
        let r = self.a * (1.0 - self.e * ecc.cos());
        let (xo, yo) = (r * nu.cos(), r * nu.sin());

        // Rotate the in-plane point into the ecliptic frame.
        let (sw, cw) = self.peri_deg.to_radians().sin_cos();
        let (so, co) = self.node_deg.to_radians().sin_cos();
        let (si, ci) = self.inc_deg.to_radians().sin_cos();

        let x = (co * cw - so * sw * ci) * xo + (-co * sw - so * cw * ci) * yo;
        let y = (so * cw + co * sw * ci) * xo + (-so * sw + co * cw * ci) * yo;
        let z = (sw * si) * xo + (cw * si) * yo;
        DVec3::new(x, y, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The solved eccentric anomaly must actually satisfy Kepler's equation.
    #[test]
    fn kepler_solution_is_consistent() {
        for &m in &[0.3, 1.7, 3.0, 5.5] {
            for &e in &[0.0, 0.05, 0.3, 0.7] {
                let ecc = solve_kepler(m, e);
                let residual = ecc - e * ecc.sin() - m.rem_euclid(std::f64::consts::TAU);
                assert!(residual.abs() < 1e-9, "M={m} e={e} residual={residual}");
            }
        }
    }

    /// A circular orbit (e = 0) keeps a constant radius equal to `a`.
    #[test]
    fn circular_orbit_has_constant_radius() {
        let el = Elements {
            a: 0.01,
            e: 0.0,
            inc_deg: 10.0,
            node_deg: 40.0,
            peri_deg: 0.0,
            m0_deg: 0.0,
            period: 5.0,
            epoch: 2_451_545.0,
        };
        for k in 0..8 {
            let jd = 2_451_545.0 + k as f64 * 0.5;
            let r = el.position(jd).length();
            assert!((r - 0.01).abs() < 1e-9, "radius drifted to {r}");
        }
    }
}
