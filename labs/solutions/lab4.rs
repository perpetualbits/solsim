// Reference solution for Lab 4 — try it yourself first!
//
// Not compiled; reference only. Replace the body of `rk4_step` in ../src/lib.rs.

pub fn rk4_step(r: DVec3, v: DVec3, gm: f64, dt: f64) -> (DVec3, DVec3) {
    // Shorthand for "the acceleration at position p".
    let a = |p: DVec3| gravity_acceleration(p, gm);

    // Four estimates of the rate of change of (position, velocity).
    let k1r = v;
    let k1v = a(r);

    let k2r = v + 0.5 * dt * k1v;
    let k2v = a(r + 0.5 * dt * k1r);

    let k3r = v + 0.5 * dt * k2v;
    let k3v = a(r + 0.5 * dt * k2r);

    let k4r = v + dt * k3v;
    let k4v = a(r + dt * k3r);

    // Combine them with the 1-2-2-1 weighting.
    let r_new = r + dt / 6.0 * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
    let v_new = v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
    (r_new, v_new)
}
