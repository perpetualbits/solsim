//! The energy bookkeeping of the planet system: kinetic + gravitational potential.
//!
//! A gravitating system has two kinds of energy: the *kinetic* energy of motion
//! and the *potential* energy stored in the pull of gravity. Newton's laws (and,
//! more deeply, the Hamiltonian — see `docs/MATHS.md`) guarantee that for a closed
//! system their **sum is constant** over time. So plotting the sum is a built-in
//! honesty check on the simulation: a flat line means the integrator conserves
//! energy; a drifting line reveals numerical error (or the exaggerated GR term,
//! which is not a true conservative force).

use glam::DVec3;

/// Kinetic and gravitational potential energy of the planets around the Sun.
///
/// What: returns `(KE, PE)` for all integrated planets, with the Sun fixed at the
/// origin.
/// How/why:
/// • Kinetic:  `KE = Σ ½·mᵢ·vᵢ²`  — the energy of motion of each planet.
/// • Potential: `PE = −Σ G·M_sun·mᵢ/rᵢ  −  Σ_{i<j} G·mᵢ·mⱼ/r_ij`
///   (the Sun's pull on each planet, plus the planets pulling on each other);
///   gravity's potential energy is negative because you would have to add energy
///   to pull the bodies apart to infinity.
/// We measure mass in **solar masses**, so `M_sun = 1` and `G = sun_gm`. Each
/// planet's mass is then `mᵢ = gmᵢ / sun_gm`, which makes the terms tidy:
/// `G·M_sun·mᵢ = gmᵢ` and `G·mᵢ·mⱼ = gmᵢ·gmⱼ / sun_gm`.
/// Principle: total energy `E = KE + PE` is conserved for gravity (it is the
/// system's Hamiltonian). The virial theorem adds that, averaged over an orbit,
/// `2·KE + PE = 0`; for a circular orbit this holds at every instant.
/// Units: `gm`/`sun_gm` in AU³·day⁻²; `pos` in AU; `vel` in AU·day⁻¹; the returned
/// energies are in M_sun·AU²·day⁻² (mass in solar masses).
pub fn system_energy(pos: &[DVec3], vel: &[DVec3], gm: &[f64], sun_gm: f64) -> (f64, f64) {
    let n = pos.len().min(vel.len()).min(gm.len());
    let mut ke = 0.0;
    let mut pe = 0.0;
    for i in 0..n {
        let m_i = gm[i] / sun_gm; // this planet's mass, in solar masses
        ke += 0.5 * m_i * vel[i].length_squared();

        // The Sun's pull (Sun fixed at the origin): G·M_sun·mᵢ = gmᵢ.
        pe -= gm[i] / pos[i].length();

        // Every other planet's pull: G·mᵢ·mⱼ = gmᵢ·gmⱼ / sun_gm.
        for j in (i + 1)..n {
            let r = (pos[j] - pos[i]).length();
            pe -= gm[i] * gm[j] / (sun_gm * r);
        }
    }
    (ke, pe)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::constants::GM_SUN;

    /// For a circular orbit the virial theorem must hold exactly: `2·KE + PE = 0`.
    ///
    /// A body on a circular orbit at radius `r` moves at `v = √(GM_sun/r)`, so
    /// `KE = ½·m·GM_sun/r` and `PE = −GM_sun·m/r`; hence `KE = −½·PE`. Checking
    /// this validates both halves of [`system_energy`] against a known result.
    #[test]
    fn circular_orbit_satisfies_virial() {
        let r = 1.0;
        let m_gm = GM_SUN / 332_946.05; // an Earth-mass planet (its G·m)
        let v = (GM_SUN / r).sqrt();
        let pos = [DVec3::new(r, 0.0, 0.0)];
        let vel = [DVec3::new(0.0, v, 0.0)];
        let (ke, pe) = system_energy(&pos, &vel, &[m_gm], GM_SUN);

        // 2·KE + PE should be zero to within rounding.
        assert!(
            (2.0 * ke + pe).abs() < 1e-15,
            "virial: 2KE+PE = {}",
            2.0 * ke + pe
        );
        // Total energy of a bound orbit is negative.
        assert!(ke + pe < 0.0, "bound orbit total energy must be negative");
    }
}
