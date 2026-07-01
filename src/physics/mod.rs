//! The numerical physics engine: gravity, the GR correction, and the integrator.
//!
//! This is the alternative to the analytic ephemeris: instead of reading where the
//! planets are, it computes their motion step by step from the forces. Keeping it
//! separate from the rendering and the ephemeris (a house rule) makes the physics
//! easy to test on its own.

pub mod energy;
pub mod forces;
pub mod galactic;
pub mod galaxy_ic;
pub mod gpu;
pub mod nbody;
pub mod octree;
pub mod particles;
