//! The orbit camera: where we look at the scene from.
//!
//! The camera floats on the surface of an imaginary sphere centred on a **target**
//! point and always looks inward at that target. Its position is given by two
//! angles — `theta` (around, like a compass bearing) and `phi` (up/down) — and a
//! `radius` (how far away). Turning the angles spins the view; changing the radius
//! zooms. This is the most natural way to inspect something from all sides.

use glam::{DMat4, DVec3, Mat4};

/// How far the up/down angle may approach the poles, in radians.
///
/// What: a small gap kept between `phi` and straight up/down.
/// How/why: if the camera looked exactly along its "up" direction the view matrix
/// would be undefined (a zero-length cross product), so we stop just short.
/// Units: radians.
const PHI_LIMIT: f64 = 1.5533; // ≈ 89°

/// An orbiting, target-looking camera.
///
/// What: stores the camera's two angles, distance, target and lens settings.
/// How/why: keeping the camera in spherical coordinates (`theta`, `phi`, `radius`)
/// makes "drag to rotate" and "wheel to zoom" trivial to implement.
/// Units: `theta`/`phi`/`fovy` in radians; `radius`/`znear`/`zfar` in AU;
/// `target` is a position in AU.
pub struct OrbitCamera {
    /// Azimuth angle (around the vertical axis), in radians.
    pub theta: f64,
    /// Elevation angle (up/down), in radians, kept within ±[`PHI_LIMIT`].
    pub phi: f64,
    /// Distance from the target, in AU.
    pub radius: f64,
    /// Closest the camera may get to the target, in AU. The app sets this each
    /// frame to just above the focused body's surface so you cannot zoom through it.
    pub min_radius: f64,
    /// The point the camera looks at, in AU (the floating-origin centre).
    pub target: DVec3,
    /// Vertical field of view, in radians.
    pub fovy: f64,
}

impl Default for OrbitCamera {
    /// Sensible starting camera for the Sun–Earth–Moon view.
    ///
    /// What: a camera placed close enough to the target to see the Moon's orbit.
    /// How/why: the default radius (0.025 AU) frames the Moon's ≈0.0026 AU orbit
    /// nicely; the angles give a gentle three-quarter view. The near/far clip
    /// planes are not stored — they are derived from the zoom in
    /// [`clip_planes`](Self::clip_planes) so you can zoom from a moon's surface out
    /// to the whole system.
    /// Units: see [`OrbitCamera`].
    fn default() -> Self {
        Self {
            theta: 0.6,
            phi: 0.5,
            radius: 0.025,
            min_radius: 1.0e-5,
            target: DVec3::ZERO,
            fovy: std::f64::consts::FRAC_PI_4, // 45°
        }
    }
}

impl OrbitCamera {
    /// Rotate the camera by dragging.
    ///
    /// What: changes the two viewing angles.
    /// How/why: we add the mouse movement (scaled to radians) to `theta` and
    /// `phi`, then clamp `phi` so the camera never reaches the poles where the
    /// view would break down.
    /// Units: `dtheta`/`dphi` in radians.
    pub fn orbit(&mut self, dtheta: f64, dphi: f64) {
        self.theta += dtheta;
        self.phi = (self.phi + dphi).clamp(-PHI_LIMIT, PHI_LIMIT);
    }

    /// Zoom in or out.
    ///
    /// What: multiplies the distance to the target by a factor.
    /// How/why: multiplying (rather than adding) makes each scroll step feel the
    /// same at every scale; the result is clamped so we cannot zoom inside the
    /// target or impossibly far away.
    /// Units: `factor` is dimensionless (e.g. 0.9 zooms in, 1.1 zooms out).
    pub fn zoom(&mut self, factor: f64) {
        self.radius = (self.radius * factor).clamp(self.min_radius, 100.0);
    }

    /// Near and far clip distances for the current zoom.
    ///
    /// What: how close and how far the camera can see, scaled to the zoom.
    /// How/why: tying both planes to the distance-to-target keeps good depth
    /// precision at every scale — when you zoom right up to a moon the near plane
    /// shrinks with you, and when you pull back the far plane grows to keep distant
    /// bodies in view. Without this, a fixed near plane would clip everything when
    /// you zoom in close.
    /// Units: AU.
    fn clip_planes(&self) -> (f64, f64) {
        let near = (self.radius * 0.002).clamp(1.0e-7, 0.5);
        let far = (self.radius * 3000.0).max(100.0);
        (near, far)
    }

    /// The camera's position relative to its target.
    ///
    /// What: the offset vector from the target to the camera (the "eye").
    /// How/why: standard spherical-to-rectangular conversion,
    /// `(r·cosφ·cosθ, r·cosφ·sinθ, r·sinφ)`, with `z` as ecliptic north; this is
    /// where the camera sits on its viewing sphere.
    /// Units: AU.
    pub fn eye_offset(&self) -> DVec3 {
        let (sp, cp) = self.phi.sin_cos();
        let (st, ct) = self.theta.sin_cos();
        self.radius * DVec3::new(cp * ct, cp * st, sp)
    }

    /// Build the combined view-projection matrix in f32 for the GPU.
    ///
    /// What: returns the matrix that turns AU positions (already shifted into the
    /// camera's floating-origin frame) into screen coordinates.
    /// How/why: we build the look-at and perspective matrices in f64 for accuracy,
    /// using the target as the origin (the "floating origin" trick — the camera
    /// looks from `eye_offset` toward `(0,0,0)`), then multiply and cast to f32.
    /// Working in f64 until the very end keeps tiny Moon-scale detail from being
    /// lost in f32 rounding.
    /// Principle: a perspective projection makes distant things look smaller, like
    /// a real camera; the look-at matrix points the camera at the target.
    /// Units: input `aspect` is width/height (dimensionless); the returned matrix
    /// is dimensionless.
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let eye = self.eye_offset();
        let view = DMat4::look_at_rh(eye, DVec3::ZERO, DVec3::Z);
        let (near, far) = self.clip_planes();
        let proj = DMat4::perspective_rh(self.fovy, aspect as f64, near, far);
        (proj * view).as_mat4()
    }

    /// Build a rotation-only view-projection matrix for the star background.
    ///
    /// What: like [`view_proj`](Self::view_proj) but with the camera fixed at the
    /// origin, so only its *orientation* matters.
    /// How/why: we look from the origin along the same direction the camera faces
    /// (toward `−eye_offset`); dropping the position means the stars never shift as
    /// you zoom or pan — they behave as if infinitely far away — while still
    /// turning correctly when you rotate the view.
    /// Principle: stars are so distant that only line-of-sight direction matters,
    /// not where in the solar system you stand.
    /// Units: `aspect` is width/height; returns a dimensionless matrix.
    pub fn star_view_proj(&self, aspect: f32) -> Mat4 {
        let forward = -self.eye_offset();
        let view = DMat4::look_at_rh(DVec3::ZERO, forward, DVec3::Z);
        let (near, far) = self.clip_planes();
        let proj = DMat4::perspective_rh(self.fovy, aspect as f64, near, far);
        (proj * view).as_mat4()
    }
}
