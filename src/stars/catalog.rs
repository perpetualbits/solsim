//! Loading the bright-star catalogue.
//!
//! We ship the 1500 brightest stars of the Yale Bright Star Catalogue (BSC5) as a
//! small CSV baked into the program, so there is no file to find at run time. Each
//! row gives a star's sky position, brightness and colour.

/// One star from the catalogue.
///
/// What: a star's name, sky position, brightness and colour index.
/// How/why: these are exactly the columns of our CSV; later code turns them into a
/// direction on the sky, a colour and a dot size.
/// Units: `ra_deg`/`dec_deg` in degrees (equatorial J2000); `vmag` is apparent
/// visual magnitude (smaller = brighter); `bv` is the B−V colour index (bigger =
/// redder).
pub struct Star {
    /// Star designation; not shown yet, but kept for future labels and the manual.
    #[allow(dead_code)]
    pub name: String,
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub vmag: f64,
    pub bv: f64,
}

/// The catalogue CSV, compiled straight into the binary.
///
/// What: the text of `assets/bsc5.csv`.
/// How/why: `include_str!` embeds the file at build time, so the program never has
/// to locate it on disk. Columns: name, ra_deg, dec_deg, vmag, bv.
/// Units: see [`Star`].
const CATALOG_CSV: &str = include_str!("../../assets/bsc5.csv");

/// Parse the built-in catalogue into a list of stars.
///
/// What: reads every data row of the embedded CSV into a [`Star`].
/// How/why: we skip the header line, split each remaining line on commas, and
/// parse the five fields; any malformed row is silently skipped rather than
/// crashing the program (so a stray line can never take down the renderer).
/// Units: see [`Star`]; returns the stars in the file's order (brightest first).
pub fn load() -> Vec<Star> {
    let mut stars = Vec::new();
    for line in CATALOG_CSV.lines().skip(1) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut fields = line.split(',');
        let name = fields.next().unwrap_or("").trim().to_string();
        let (Some(ra), Some(dec), Some(vmag), Some(bv)) = (
            fields.next().and_then(|s| s.trim().parse::<f64>().ok()),
            fields.next().and_then(|s| s.trim().parse::<f64>().ok()),
            fields.next().and_then(|s| s.trim().parse::<f64>().ok()),
            fields.next().and_then(|s| s.trim().parse::<f64>().ok()),
        ) else {
            continue;
        };
        stars.push(Star {
            name,
            ra_deg: ra,
            dec_deg: dec,
            vmag,
            bv,
        });
    }
    stars
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The catalogue should load a full sky of stars led by Sirius.
    #[test]
    fn loads_catalogue() {
        let stars = load();
        assert!(stars.len() > 1000, "only {} stars loaded", stars.len());
        // The list is sorted brightest-first, so the first star is Sirius.
        let sirius = &stars[0];
        assert!(sirius.vmag < -1.0, "brightest vmag was {}", sirius.vmag);
        assert!((sirius.ra_deg - 101.29).abs() < 0.5);
        assert!((sirius.dec_deg + 16.72).abs() < 0.5);
    }
}
