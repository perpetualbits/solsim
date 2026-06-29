// Reference solution for Lab 2 — try it yourself first!
//
// This file is NOT compiled; it is here only to check your answer. Replace the
// body of `gravity_acceleration` in ../src/lib.rs with the lines below.

pub fn gravity_acceleration(r: DVec3, gm: f64) -> DVec3 {
    // |r⃗|, the distance to the Sun at the origin.
    let len = r.length();
    // a⃗ = −G·M · r⃗ / |r⃗|³  (inverse-square strength × direction toward the Sun).
    -gm * r / (len * len * len)
}
