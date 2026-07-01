//! A tiny deterministic pseudo-random generator (SplitMix64).
//!
//! Used wherever we want *reproducible* procedural content — the Milky Way band,
//! the neighbour galaxies, and the colliding-galaxy initial conditions — without
//! pulling in a random-number crate. Same seed ⇒ same stream ⇒ same result every
//! run. It is emphatically not for cryptography; it is for scattering points.

/// A seeded SplitMix64 generator.
///
/// What: holds the 64-bit state; each draw advances and hashes it.
/// How/why: adding the golden-ratio constant then mixing the bits gives a stream
/// that looks independent — plenty good for placing particles.
/// Units: none.
pub struct Rng(u64);

impl Rng {
    /// Start the generator from a fixed seed.
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    /// Return the next 64 random bits (one SplitMix64 step).
    ///
    /// What: advances the state and hashes it.
    /// How/why: add the golden-ratio constant, then two xor-shift-multiply rounds
    /// scramble the bits so consecutive outputs look independent.
    /// Units: none (raw bits).
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform random number in the half-open interval [0, 1).
    ///
    /// What: a fractional random value.
    /// How/why: take the top 53 bits (the `f64` mantissa width) and divide by 2⁵³.
    /// Units: dimensionless.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A standard-normal random number (mean 0, standard deviation 1).
    ///
    /// What: a bell-curve-distributed value.
    /// How/why: the Box–Muller transform turns two uniforms into a normal value
    /// `√(−2·ln u₁)·cos(2π·u₂)`.
    /// Units: dimensionless (scale by σ for a chosen spread).
    pub fn gaussian(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}
