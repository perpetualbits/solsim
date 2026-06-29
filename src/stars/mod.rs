//! The bright-star background: catalogue, colours and sky placement.
//!
//! These modules are pure data and maths — they turn the Yale Bright Star
//! Catalogue into coloured, sized points placed on the sky. The actual drawing
//! lives in `render::starfield`.

pub mod catalog;
pub mod color;
pub mod galaxy;
pub mod project;
