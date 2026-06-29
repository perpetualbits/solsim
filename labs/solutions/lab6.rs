// Reference solution for Lab 6 — try it yourself first!
//
// Not compiled; reference only. Replace the body of `solve_kepler` in ../src/lib.rs.

pub fn solve_kepler(mean_anomaly: f64, eccentricity: f64) -> f64 {
    // Start with a sensible guess: for small e, E is close to M.
    let mut big_e = mean_anomaly;
    // Newton's method: improve the guess a few times. Each round roughly doubles
    // the number of correct digits, so 8 rounds is far more than enough.
    for _ in 0..8 {
        let f = big_e - eccentricity * big_e.sin() - mean_anomaly; // should reach 0
        let f_prime = 1.0 - eccentricity * big_e.cos(); // its slope
        big_e -= f / f_prime;
    }
    big_e
}
