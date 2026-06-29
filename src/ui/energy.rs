//! The energy overlay: a small line graph of kinetic, potential and total energy.
//!
//! It plots the three numbers produced by [`crate::physics::energy::system_energy`]
//! as they change over time. The point for a student is the **total** (white) line:
//! it should stay flat, because energy is conserved. If it visibly slopes or
//! wobbles, that is numerical drift — or the exaggerated General-Relativity term,
//! which is not a true conservative force. The graph draws itself with egui's
//! painter, so it needs no extra plotting library.

use egui::{pos2, Color32, Sense, Stroke, Vec2};

/// One recorded moment of the system's energy.
///
/// What: kinetic, potential and total energy at one instant.
/// How/why: the app pushes one of these per simulated step into a rolling buffer,
/// and this module draws the buffer as three lines.
/// Units: M_sun·AU²·day⁻² (see [`crate::physics::energy`]).
#[derive(Clone, Copy)]
pub struct Sample {
    pub ke: f64,
    pub pe: f64,
    pub total: f64,
}

/// Colours for the three lines (linear-ish sRGB, chosen to read on a dark panel).
const KE_COLOR: Color32 = Color32::from_rgb(90, 220, 120); // green  — kinetic
const PE_COLOR: Color32 = Color32::from_rgb(235, 130, 70); // orange — potential
const TOTAL_COLOR: Color32 = Color32::from_rgb(240, 240, 245); // white — sum

/// Draw the energy graph window if it is open.
///
/// What: shows the KE/PE/total history as a self-scaling line graph plus current
/// numbers and the total-energy drift.
/// How/why: it finds the smallest and largest value across all three series so the
/// curves fill the canvas, maps each sample to a pixel, and draws three polylines;
/// a faint line marks zero. The "drift" is `(max−min)` of the total line over the
/// window, shown as a fraction of the typical energy so the student can judge how
/// well energy is conserved.
/// Units: energies in M_sun·AU²·day⁻²; drift is dimensionless (a ratio).
pub fn show(ctx: &egui::Context, open: &mut bool, hist: &[Sample]) {
    if !*open {
        return;
    }
    let mut keep_open = true;
    egui::Window::new("Energy (KE + PE = constant?)")
        .open(&mut keep_open)
        .default_size([380.0, 250.0])
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(
                "Kinetic + potential energy of the planets. The white 'total' line \
should stay flat — energy is conserved.",
            );

            let canvas = Vec2::new(360.0, 170.0);
            let (rect, _) = ui.allocate_exact_size(canvas, Sense::hover());
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 3.0, Color32::from_rgb(18, 18, 24));

            if hist.len() < 2 {
                painter.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "collecting data… (let time run)",
                    egui::FontId::proportional(13.0),
                    Color32::GRAY,
                );
                return;
            }

            // Find the value range across all three series, with a little padding,
            // so the curves use the whole canvas height.
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for s in hist {
                for v in [s.ke, s.pe, s.total] {
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            if hi <= lo {
                hi = lo + 1.0;
            }
            let pad = (hi - lo) * 0.08;
            lo -= pad;
            hi += pad;

            let n = hist.len();
            let x_of = |i: usize| rect.left() + rect.width() * (i as f32 / (n - 1) as f32);
            let y_of = |v: f64| rect.bottom() - rect.height() * ((v - lo) / (hi - lo)) as f32;

            // Zero reference line, if zero falls inside the visible range.
            if lo < 0.0 && hi > 0.0 {
                let y = y_of(0.0);
                painter.line_segment(
                    [pos2(rect.left(), y), pos2(rect.right(), y)],
                    Stroke::new(1.0, Color32::from_gray(70)),
                );
            }

            // The three lines. Total is drawn last (on top) and a touch thicker.
            for (pick, color, width) in [
                (0usize, KE_COLOR, 1.4),
                (1, PE_COLOR, 1.4),
                (2, TOTAL_COLOR, 2.0),
            ] {
                let pts: Vec<egui::Pos2> = (0..n)
                    .map(|i| {
                        let s = hist[i];
                        let v = [s.ke, s.pe, s.total][pick];
                        pos2(x_of(i), y_of(v))
                    })
                    .collect();
                painter.add(egui::Shape::line(pts, Stroke::new(width, color)));
            }

            // Legend and current numbers.
            let last = hist[n - 1];
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.colored_label(KE_COLOR, format!("KE {:.3e}", last.ke));
                ui.colored_label(PE_COLOR, format!("PE {:.3e}", last.pe));
                ui.colored_label(TOTAL_COLOR, format!("Total {:.3e}", last.total));
            });

            // Drift of the total line over the window, as a fraction of its size.
            let mut tlo = f64::INFINITY;
            let mut thi = f64::NEG_INFINITY;
            for s in hist {
                tlo = tlo.min(s.total);
                thi = thi.max(s.total);
            }
            let scale = last.total.abs().max(1e-300);
            let drift = (thi - tlo) / scale;
            ui.label(format!(
                "Total drift over window: {drift:.2e} (relative). Smaller is better; \
the GR engine and big steps make it grow."
            ));
            ui.label("Press Y to close. Units: M_sun·AU²/day².");
        });
    // Honour the window's own close button as well as the Y key.
    *open = keep_open;
}
