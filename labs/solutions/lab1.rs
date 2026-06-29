// Reference solution for Lab 1 — try it yourself first!
//
// This file is NOT compiled; it is here only to check your answer. Replace the
// body of `advance_position` in ../src/lib.rs with the one line below.

pub fn advance_position(r: DVec3, v: DVec3, dt: f64) -> DVec3 {
    // The body moves by velocity × time (a vector), so:
    r + v * dt
}
