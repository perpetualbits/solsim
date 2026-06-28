//! Optional start-up configuration, read from a simple text file.
//!
//! If a file `solarsim.conf` sits next to the program (the current directory), its
//! settings override the defaults. The format is one `key = value` per line, with
//! `#` for comments — no extra libraries needed. Everything has a sensible default,
//! so the file is entirely optional.

/// All the settings the program reads at start-up.
///
/// What: the observer's place, the starting date and speed, the trail length and
/// the window size.
/// How/why: bundling them keeps start-up tidy; each field is filled from the file
/// if present, otherwise from [`Config::default`].
/// Units: latitude/longitude in degrees; `speed_days_per_sec` in simulated days
/// per real second; `trail_length` a count of points; window size in pixels;
/// `start_date` is `(year, month, day)` or `None` for "today".
pub struct Config {
    pub observer_lat: f64,
    pub observer_lon: f64,
    pub speed_days_per_sec: f64,
    pub trail_length: usize,
    pub window_width: u32,
    pub window_height: u32,
    pub start_date: Option<(i16, u8, u8)>,
}

impl Default for Config {
    /// The built-in defaults (used when there is no config file).
    ///
    /// What: Zutphen, a few days/second, a 4000-point trail, a 1280×800 window,
    /// starting today.
    /// How/why: these match the values the simulator used before configuration
    /// existed, so behaviour is unchanged without a file.
    /// Units: see [`Config`].
    fn default() -> Self {
        Self {
            observer_lat: 52.14,
            observer_lon: 6.20,
            speed_days_per_sec: 4.63, // ≈ 400000× real time
            trail_length: 4000,
            window_width: 1280,
            window_height: 800,
            start_date: None,
        }
    }
}

/// Load the configuration, applying any `solarsim.conf` found next to the program.
///
/// What: returns a [`Config`] — defaults, with file values layered on top.
/// How/why: we read `solarsim.conf` from the current directory if it exists, parse
/// each `key = value` line, and overwrite the matching field. Unknown keys and
/// unparseable values are ignored (the program never crashes on a bad config).
/// Units: see [`Config`].
pub fn load() -> Config {
    let mut config = Config::default();
    let Ok(text) = std::fs::read_to_string("solarsim.conf") else {
        return config;
    };

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let (key, value) = (key.trim(), value.trim());
        match key {
            "observer_lat" => set_f64(value, &mut config.observer_lat),
            "observer_lon" => set_f64(value, &mut config.observer_lon),
            "speed_days_per_sec" => set_f64(value, &mut config.speed_days_per_sec),
            "trail_length" => {
                if let Ok(v) = value.parse() {
                    config.trail_length = v;
                }
            }
            "window_width" => {
                if let Ok(v) = value.parse() {
                    config.window_width = v;
                }
            }
            "window_height" => {
                if let Ok(v) = value.parse() {
                    config.window_height = v;
                }
            }
            "start_date" => config.start_date = parse_date(value).or(config.start_date),
            _ => {}
        }
    }
    config
}

/// Overwrite a float setting if the text parses.
///
/// What: parses `value` and, only on success, stores it in `slot`.
/// How/why: keeps the `load` match arms short and makes a bad value a no-op rather
/// than a crash.
/// Units: whatever `slot` represents.
fn set_f64(value: &str, slot: &mut f64) {
    if let Ok(v) = value.parse() {
        *slot = v;
    }
}

/// Parse a `YYYY-MM-DD` date.
///
/// What: turns a date string into `(year, month, day)`.
/// How/why: splits on `-` and parses the three parts; returns `None` if the format
/// is wrong, so a typo just falls back to the default start date.
/// Units: calendar fields.
fn parse_date(value: &str) -> Option<(i16, u8, u8)> {
    let mut parts = value.split('-');
    let y = parts.next()?.trim().parse().ok()?;
    let m = parts.next()?.trim().parse().ok()?;
    let d = parts.next()?.trim().parse().ok()?;
    Some((y, m, d))
}
