// Reference solution for Lab 5 — try it yourself first!
//
// Not compiled; reference only. Replace the body of `energies` in ../src/lib.rs.

pub fn energies(mass: f64, r: DVec3, v: DVec3, gm_sun: f64) -> (f64, f64) {
    // Kinetic: ½·m·v²  (v² is the squared speed).
    let kinetic = 0.5 * mass * v.length_squared();
    // Potential: −G·M·m / r  (negative — gravity is a bound well).
    let potential = -gm_sun * mass / r.length();
    (kinetic, potential)
}
