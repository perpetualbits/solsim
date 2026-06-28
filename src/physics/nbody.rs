//! The N-body state and the RK4 integrator that moves it forward in time.
//!
//! Instead of reading positions from a formula (the ephemeris), this *computes*
//! them: it knows each planet's position and velocity, works out the forces, and
//! steps everything forward in tiny time steps. This is what lets us switch on the
//! General-Relativity term and watch an orbit slowly precess.

use glam::DVec3;

use super::forces;

/// Position and velocity of every integrated body (the planets).
///
/// What: the simulation's moving state — where each planet is and how fast it is
/// going.
/// How/why: a numerical integrator needs both, because gravity sets the
/// acceleration, which changes velocity, which changes position.
/// Units: `pos` in AU, `vel` in AU/day; the two vectors line up by index.
pub struct State {
    pub pos: Vec<DVec3>,
    pub vel: Vec<DVec3>,
}

/// Advance the state by one time step using 4th-order Runge–Kutta.
///
/// What: moves all bodies forward by `dt` days.
/// How/why: RK4 samples the acceleration four times across the step (start, two
/// midpoints, end) and combines them with weights 1-2-2-1; this cancels low-order
/// error so the orbits stay accurate with far larger steps than a naive method.
/// The state is `y = (pos, vel)` and its rate of change is `(vel, acceleration)`.
/// Principle: a planet's path is the solution of `r¨ = a(r, v)`; RK4 follows that
/// curve closely by averaging several tangent estimates.
/// Units: `gm`/`sun_gm` in AU³·day⁻²; `gr_strength` dimensionless; `dt` in days.
pub fn rk4_step(state: &mut State, gm: &[f64], sun_gm: f64, gr_strength: f64, dt: f64) {
    let n = state.pos.len();
    let accel = |p: &[DVec3], v: &[DVec3]| forces::accelerations(p, v, gm, sun_gm, gr_strength);

    // k1: rate at the start of the step.
    let k1p = state.vel.clone();
    let k1v = accel(&state.pos, &state.vel);

    // k2: rate at the first midpoint.
    let p2: Vec<DVec3> = (0..n).map(|i| state.pos[i] + 0.5 * dt * k1p[i]).collect();
    let v2: Vec<DVec3> = (0..n).map(|i| state.vel[i] + 0.5 * dt * k1v[i]).collect();
    let k2p = v2.clone();
    let k2v = accel(&p2, &v2);

    // k3: rate at the second midpoint.
    let p3: Vec<DVec3> = (0..n).map(|i| state.pos[i] + 0.5 * dt * k2p[i]).collect();
    let v3: Vec<DVec3> = (0..n).map(|i| state.vel[i] + 0.5 * dt * k2v[i]).collect();
    let k3p = v3.clone();
    let k3v = accel(&p3, &v3);

    // k4: rate at the end of the step.
    let p4: Vec<DVec3> = (0..n).map(|i| state.pos[i] + dt * k3p[i]).collect();
    let v4: Vec<DVec3> = (0..n).map(|i| state.vel[i] + dt * k3v[i]).collect();
    let k4p = v4.clone();
    let k4v = accel(&p4, &v4);

    // Combine the four estimates with the 1-2-2-1 weighting.
    for i in 0..n {
        state.pos[i] += dt / 6.0 * (k1p[i] + 2.0 * k2p[i] + 2.0 * k3p[i] + k4p[i]);
        state.vel[i] += dt / 6.0 * (k1v[i] + 2.0 * k2v[i] + 2.0 * k3v[i] + k4v[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::constants::GM_SUN;

    /// The GR term must advance the perihelion (Newtonian must not).
    ///
    /// We integrate a Mercury-like orbit for one period and measure how far the
    /// eccentricity vector (which points at perihelion) turns. With the GR term off
    /// it should barely move (closed ellipse); with it on, it should turn by about
    /// `gr_strength × 6π·μ/(c²·a·(1−e²))` — the textbook advance, exaggerated.
    #[test]
    fn gr_advances_perihelion() {
        use crate::astro::constants::C_LIGHT;
        let mu = GM_SUN;
        let a = 0.387099;
        let e = 0.205630;

        // Start at perihelion, moving perpendicular to the Sun line.
        let rp = a * (1.0 - e);
        let vp = (mu * (1.0 + e) / (a * (1.0 - e))).sqrt();
        let period = std::f64::consts::TAU * (a * a * a / mu).sqrt();

        // Eccentricity vector: points toward perihelion; its angle is ϖ.
        let evec = |p: DVec3, v: DVec3| {
            let h = p.cross(v);
            v.cross(h) / mu - p.normalize()
        };
        let angle = |p: DVec3, v: DVec3| {
            let ev = evec(p, v);
            ev.y.atan2(ev.x)
        };

        let run = |gr: f64| {
            let mut s = State {
                pos: vec![DVec3::new(rp, 0.0, 0.0)],
                vel: vec![DVec3::new(0.0, vp, 0.0)],
            };
            let a0 = angle(s.pos[0], s.vel[0]);
            let steps = 20_000;
            let h = period / steps as f64;
            for _ in 0..steps {
                rk4_step(&mut s, &[0.0], mu, gr, h);
            }
            angle(s.pos[0], s.vel[0]) - a0
        };

        // Newtonian: essentially closed.
        assert!(run(0.0).abs() < 1e-5, "Newtonian advance {}", run(0.0));

        // Relativistic (exaggerated): matches the closed-form prediction.
        let gr = 2000.0;
        let expected = gr * 6.0 * std::f64::consts::PI * mu
            / (C_LIGHT * C_LIGHT * a * (1.0 - e * e));
        let measured = run(gr);
        assert!(measured > 0.0, "GR advance should be positive, got {measured}");
        assert!(
            (measured - expected).abs() < 0.15 * expected,
            "GR advance {measured} vs expected {expected}"
        );
    }

    /// A body on a circular orbit should keep a constant distance from the Sun.
    #[test]
    fn circular_orbit_stays_circular() {
        let a = 1.0;
        // Circular speed at radius a around the Sun: v = sqrt(GM_sun / a).
        let v = (GM_SUN / a).sqrt();
        let mut state = State {
            pos: vec![DVec3::new(a, 0.0, 0.0)],
            vel: vec![DVec3::new(0.0, v, 0.0)],
        };
        // No other bodies, no GR: a pure two-body circular orbit.
        for _ in 0..400 {
            rk4_step(&mut state, &[0.0], GM_SUN, 0.0, 1.0);
            let r = state.pos[0].length();
            assert!((r - a).abs() < 1e-3, "radius drifted to {r}");
        }
    }
}
