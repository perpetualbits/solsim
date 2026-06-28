//! The astronomy core: time-keeping, constants, and body positions.
//!
//! These modules contain only pure maths — no graphics and no GPU. Keeping the
//! physics separate from the rendering (a project house rule) makes each part
//! easy to read and to test against known values.

// This module is built up phase by phase: some constants and helpers defined now
// (e.g. the speed-of-light, the body table, the velocity helper) are only consumed
// by later phases. Allow dead code here so the early-phase build stays warning-free
// without hiding warnings in the rest of the program.
#![allow(dead_code)]

pub mod constants;
pub mod elements;
pub mod ephemeris;
pub mod time;
