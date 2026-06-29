// Reference solution for Lab 3 — try it yourself first!
//
// Not compiled; reference only. Replace the body of `euler_step` in ../src/lib.rs.

pub fn euler_step(r: DVec3, v: DVec3, gm: f64, dt: f64) -> (DVec3, DVec3) {
    // Acceleration from gravity at the start of the step (Lab 2).
    let a = gravity_acceleration(r, gm);
    // Forward Euler: position uses the OLD velocity; velocity uses the pull.
    let r_new = r + v * dt;
    let v_new = v + a * dt;
    (r_new, v_new)
}
