//! Turning a star's colour index and brightness into a colour and a dot size.
//!
//! A star's colour comes from its temperature, which we get from its B−V colour
//! index; its drawn size comes from its brightness (magnitude). Hot stars look
//! blue-white, cool stars orange-red; brighter stars are drawn as bigger dots.

/// Reference magnitude that maps to the base dot size.
const M_REF: f64 = 0.0;
/// Base dot size (pixels) for a star of magnitude [`M_REF`].
const BASE_SIZE: f32 = 10.5;
/// Smallest and largest dot sizes, in pixels.
const MIN_SIZE: f32 = 3.6;
const MAX_SIZE: f32 = 48.0;

/// Estimate a star's surface temperature from its B−V colour index.
///
/// What: returns the temperature in kelvin.
/// How/why: the Ballesteros (2012) formula
/// `T = 4600·(1/(0.92·(B−V)+1.7) + 1/(0.92·(B−V)+0.62))` fits a star's colour to
/// its temperature, treating the star roughly as a black body. A blue star
/// (B−V ≈ 0) comes out around 10000 K, a red one (B−V ≈ 1.5) around 3800 K.
/// Principle: hotter things glow bluer — the same reason a flame's blue part is
/// hottest.
/// Units: input `bv` is dimensionless (magnitudes); output in kelvin.
pub fn bv_to_temperature(bv: f64) -> f64 {
    4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62))
}

/// Convert a black-body temperature to a display RGB colour.
///
/// What: returns linear RGB in 0..1 for a glowing body at temperature `kelvin`.
/// How/why: Tanner Helland's well-known piecewise fit to black-body colours —
/// each channel is a simple curve of the temperature (in hundreds of kelvin),
/// clamped to the valid range. It captures "blue when hot, red when cool".
/// Principle: a hot object emits light across all colours but peaks bluer as it
/// heats up (Planck's law / Wien's displacement).
/// Units: input in kelvin; output is three values in 0..1.
pub fn temperature_to_rgb(kelvin: f64) -> [f32; 3] {
    let t = kelvin.clamp(1000.0, 40000.0) / 100.0;

    let red = if t <= 66.0 {
        255.0
    } else {
        329.698_727_446 * (t - 60.0).powf(-0.133_204_759_2)
    };
    let green = if t <= 66.0 {
        99.470_802_586_1 * t.ln() - 161.119_568_166_1
    } else {
        288.122_169_528_3 * (t - 60.0).powf(-0.075_514_849_2)
    };
    let blue = if t >= 66.0 {
        255.0
    } else if t <= 19.0 {
        0.0
    } else {
        138.517_731_223_1 * (t - 10.0).ln() - 305.044_792_730_7
    };

    [
        (red.clamp(0.0, 255.0) / 255.0) as f32,
        (green.clamp(0.0, 255.0) / 255.0) as f32,
        (blue.clamp(0.0, 255.0) / 255.0) as f32,
    ]
}

/// Get a star's display colour directly from its B−V index.
///
/// What: combines the two steps above (colour index → temperature → RGB).
/// How/why: convenience wrapper used when building the star instances.
/// Units: input dimensionless; output RGB in 0..1.
pub fn bv_to_rgb(bv: f64) -> [f32; 3] {
    temperature_to_rgb(bv_to_temperature(bv))
}

/// Choose a star's drawn dot size from its apparent magnitude.
///
/// What: returns the dot diameter in pixels.
/// How/why: magnitude is logarithmic and inverted (smaller = brighter), so each
/// step of 5 magnitudes is a factor of 100 in brightness. We use
/// `size = BASE·10^(0.2·(M_REF − m))`, i.e. brighter → bigger, then clamp so the
/// faintest stars stay visible dots and the brightest do not become blobs.
/// Principle: the magnitude scale `m₁ − m₂ = −2.5·log₁₀(F₁/F₂)` inverted gives the
/// `10^(0.2·…)` factor.
/// Units: input `vmag` in magnitudes; output in pixels.
pub fn magnitude_to_size(vmag: f64) -> f32 {
    let size = BASE_SIZE * 10f32.powf(0.2 * (M_REF - vmag) as f32);
    size.clamp(MIN_SIZE, MAX_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// B−V → temperature should land in the right ballpark for blue and red stars.
    #[test]
    fn temperature_ballpark() {
        let hot = bv_to_temperature(0.0); // A0 star, blue-white
        assert!((9000.0..=11000.0).contains(&hot), "blue star T = {hot}");
        let cool = bv_to_temperature(1.5); // M star, orange-red
        assert!((3400.0..=4000.0).contains(&cool), "red star T = {cool}");
    }

    /// Hot stars should be bluer than cool stars (more blue than red).
    #[test]
    fn hot_is_bluer_than_cool() {
        let hot = temperature_to_rgb(10000.0);
        let cool = temperature_to_rgb(3500.0);
        assert!(hot[2] > hot[0] * 0.8, "hot star not blue enough: {hot:?}");
        assert!(cool[0] > cool[2], "cool star not red enough: {cool:?}");
    }

    /// Brighter stars (smaller magnitude) must be drawn larger.
    #[test]
    fn brighter_is_bigger() {
        assert!(magnitude_to_size(-1.0) > magnitude_to_size(3.0));
    }
}
