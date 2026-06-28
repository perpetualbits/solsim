//! Keeping track of *when* the simulation is, in Julian Dates.
//!
//! A **Julian Date (JD)** is a single number counting days (and fractions of a
//! day) since a fixed moment far in the past. Using one number for time makes the
//! astronomy maths easy: to step forward you simply add days. The reference
//! moment [`J2000`] (1 January 2000, 12:00) has the round value 2451545.0.

use std::time::{SystemTime, UNIX_EPOCH};

use astro::time::{self, CalType, Date};

/// The Julian Date of the J2000.0 epoch: 2000-01-01 12:00 TT.
///
/// What: the standard reference instant used throughout astronomy.
/// How/why: many formulas measure time as "days since J2000"; pinning this value
/// (2451545.0) lets us check our date conversions against a known answer.
/// Units: days (a Julian Date).
pub const J2000: f64 = 2_451_545.0;

/// The Julian Date of the Unix epoch: 1970-01-01 00:00 UTC.
///
/// What: the JD value the computer's clock counts from.
/// How/why: the operating system gives time as seconds since 1970-01-01 00:00;
/// that instant is JD 2440587.5, so we can turn "seconds since 1970" into a JD by
/// adding it.
/// Units: days (a Julian Date).
const JD_UNIX_EPOCH: f64 = 2_440_587.5;

/// Number of seconds in one day.
///
/// What: the factor that converts seconds to days.
/// How/why: time in the simulation is measured in days, but real elapsed time and
/// the system clock are in seconds, so we divide by this to convert.
/// Units: seconds per day.
const SECONDS_PER_DAY: f64 = 86_400.0;

/// The simulation clock: which instant we are showing, and how fast time runs.
///
/// What: holds the current Julian Date and a speed multiplier.
/// How/why: each rendered frame we advance `jd` by the real time elapsed times
/// `speed_factor`, so a large factor makes planets race around their orbits.
/// Units: `jd` in days (a Julian Date); `speed_factor` is dimensionless
/// (simulated seconds per real second).
pub struct SimClock {
    jd: f64,
    speed_factor: f64,
}

impl SimClock {
    /// Make a clock starting at a given Julian Date.
    ///
    /// What: builds a [`SimClock`] at the supplied instant, running at 1×.
    /// How/why: we store the date and set the speed factor to 1.0 so that, until
    /// changed, one real second advances the simulation by one second.
    /// Units: `jd` in days; returns a `SimClock`.
    pub fn new(jd: f64) -> Self {
        Self {
            jd,
            speed_factor: 1.0,
        }
    }

    /// Make a clock set to the computer's current date and time.
    ///
    /// What: builds a [`SimClock`] at "now".
    /// How/why: we read the system clock as seconds since 1970, convert that to a
    /// Julian Date with [`jd_now`], and start there.
    /// Units: none in; returns a `SimClock` whose `jd` is in days.
    pub fn now() -> Self {
        Self::new(jd_now())
    }

    /// The current Julian Date of the clock.
    ///
    /// What: reads out the instant the clock is showing.
    /// How/why: a simple getter so other modules (e.g. the ephemeris) can ask
    /// "what time is it?" without touching the field directly.
    /// Units: days (a Julian Date).
    pub fn jd(&self) -> f64 {
        self.jd
    }

    /// The current speed multiplier.
    ///
    /// What: reads out how many simulated seconds pass per real second.
    /// How/why: a getter used by the on-screen display and the time controls.
    /// Units: dimensionless (simulated seconds per real second).
    pub fn speed_factor(&self) -> f64 {
        self.speed_factor
    }

    /// Set the speed multiplier.
    ///
    /// What: changes how fast simulated time flows.
    /// How/why: the time-control keys (added in Phase 6) call this; a factor of
    /// 86400 means one real second equals one simulated day.
    /// Units: dimensionless (simulated seconds per real second).
    pub fn set_speed_factor(&mut self, speed_factor: f64) {
        self.speed_factor = speed_factor;
    }

    /// Jump the clock to a specific Julian Date.
    ///
    /// What: overwrites the current instant.
    /// How/why: used to reset to "now" or to a chosen date; it just stores the
    /// new value.
    /// Units: `jd` in days.
    pub fn set_jd(&mut self, jd: f64) {
        self.jd = jd;
    }

    /// Reset the clock to the computer's current date and time.
    ///
    /// What: moves the clock to "now".
    /// How/why: convenience wrapper around [`jd_now`], bound to the `T` key later.
    /// Units: none.
    pub fn reset_to_now(&mut self) {
        self.jd = jd_now();
    }

    /// Advance the clock by a span of real time.
    ///
    /// What: moves simulated time forward by `real_seconds × speed_factor`.
    /// How/why: real seconds are turned into simulated days by multiplying by the
    /// speed factor and dividing by the seconds-in-a-day; adding that to `jd` steps
    /// the simulation. This is the heart of the animation loop.
    /// Units: `real_seconds` in seconds; the clock's `jd` advances in days.
    pub fn advance(&mut self, real_seconds: f64) {
        self.jd += real_seconds * self.speed_factor / SECONDS_PER_DAY;
    }
}

/// Convert a civil calendar date and time to a Julian Date.
///
/// What: turns a year/month/day and clock time into a single JD number.
/// How/why: we build the day-of-month as a *decimal* day
/// `day + (hr + min/60 + sec/3600)/24` (computing this ourselves so minutes and
/// seconds are scaled correctly), then hand it to the `astro` crate, which applies
/// Meeus' standard formula for the Gregorian calendar. As a check, 2000-01-01
/// 12:00 gives exactly [`J2000`] = 2451545.0.
/// Units: inputs are calendar fields (year, month 1–12, day 1–31, hours, minutes,
/// seconds); output is days (a Julian Date).
pub fn jd_from_calendar(year: i16, month: u8, day: u8, hr: u8, min: u8, sec: f64) -> f64 {
    let decimal_day = day as f64 + (hr as f64 + min as f64 / 60.0 + sec / 3600.0) / 24.0;
    let date = Date {
        year,
        month,
        decimal_day,
        cal_type: CalType::Gregorian,
    };
    time::julian_day(&date)
}

/// Convert a Julian Date back to a civil calendar date.
///
/// What: turns a JD number into year, month and decimal day.
/// How/why: the inverse of [`jd_from_calendar`]; the `astro` crate runs Meeus'
/// reverse algorithm. It returns an error for negative JDs (dates before 4713 BC),
/// which we never use, but we pass the error along instead of crashing.
/// Units: input in days (a Julian Date); output is (year, month 1–12, decimal day).
pub fn calendar_from_jd(jd: f64) -> Result<(i16, u8, f64), &'static str> {
    time::date_frm_julian_day(jd)
}

/// Read the computer's clock as a Julian Date.
///
/// What: gives the JD of the real "now".
/// How/why: we ask the OS for seconds since 1970-01-01, convert to days, and add
/// the Unix-epoch JD ([`JD_UNIX_EPOCH`]). If the clock is somehow set before 1970
/// the subtraction would fail, so we fall back to [`J2000`] rather than crash.
/// (Note: the system clock is UTC, which differs from TT by ~69 s — far too small
/// to matter at this stage.)
/// Units: returns days (a Julian Date).
pub fn jd_now() -> f64 {
    let seconds_since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    JD_UNIX_EPOCH + seconds_since_epoch / SECONDS_PER_DAY
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2000-01-01 12:00 must give exactly the J2000 epoch, 2451545.0.
    #[test]
    fn j2000_epoch_is_correct() {
        let jd = jd_from_calendar(2000, 1, 1, 12, 0, 0.0);
        assert!((jd - J2000).abs() < 1e-6, "got {jd}, expected {J2000}");
    }

    /// Converting a JD to a calendar date and back returns the same JD.
    #[test]
    fn round_trip_calendar() {
        let jd = jd_from_calendar(2026, 6, 28, 0, 0, 0.0);
        let (y, m, d) = calendar_from_jd(jd).expect("valid JD");
        assert_eq!(y, 2026);
        assert_eq!(m, 6);
        assert!((d - 28.0).abs() < 1e-6, "decimal day was {d}");
    }

    /// Advancing 1 real second at 86400× speed adds exactly one day.
    #[test]
    fn advance_scales_with_speed() {
        let mut clock = SimClock::new(J2000);
        clock.set_speed_factor(86_400.0);
        clock.advance(1.0);
        assert!((clock.jd() - (J2000 + 1.0)).abs() < 1e-9);
    }
}
