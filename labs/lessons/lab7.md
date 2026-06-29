# Practicum 7 (uitdaging) — Mercurius, Einstein en de 43″

**Doel:** een beroemde voorspelling van de algemene relativiteitstheorie
reproduceren.
**Jij past aan:** `perihelion_advance_arcsec_per_century` in `src/lib.rs`.
**Draai:** `cargo test --test lab7`

---

## Het verhaal

Newtons zwaartekracht zegt dat één enkele planeet voor altijd *dezelfde* ellips
beschrijft. Maar astronomen maten dat de ellips van Mercurius langzaam **draait**:
zijn perihelium (het punt het dichtst bij de zon) kruipt elke baan een klein stukje
verder. Het grootste deel wordt veroorzaakt door de andere planeten, maar er bleef
een hardnekkige **43 boogseconden per eeuw** onverklaard — tot Einsteins algemene
relativiteitstheorie precies dat bedrag voorspelde. Het was een van de eerste grote
bevestigingen van de theorie.

(Een boogseconde is 1/3600 van een graad — een zeer kleine hoek. 43″/eeuw is ruwweg
de breedte van een muntje gezien vanaf 100 meter, opgebouwd over honderd jaar.)

## De formule

De algemene relativiteitstheorie voegt een kleine extra trekkracht toe. Uitgewerkt
draait die elke baan over

```
Δϖ = 6π·G·M / (c²·a·(1 − e²))      (radialen per baan)
```

waarbij `a` de grootte van de baan is (de halve lange as), `e` de excentriciteit, en
`c` de lichtsnelheid. Om boogseconden per eeuw te krijgen, vermenigvuldig je met het
aantal banen in een eeuw en reken je radialen om naar boogseconden:

```
snelheid = Δϖ × (36525 / period_days) × (180·3600 / π)
```

## Wat je moet doen

```rust
use std::f64::consts::PI;
let per_orbit = 6.0 * PI * gm_sun / (C_LIGHT * C_LIGHT * a * (1.0 - e * e));
let orbits_per_century = 36_525.0 / period_days;
let rad_to_arcsec = 180.0 * 3600.0 / PI;
per_orbit * orbits_per_century * rad_to_arcsec
```

(`gm_sun = G·M`, en `C_LIGHT` wordt door de crate geleverd.)

## Controleer jezelf

```text
cargo test --test lab7
```

Mercurius (`a = 0,387099`, `e = 0,205630`, `period = 87,969` dagen) zou rond de
**43″/eeuw** moeten uitkomen; de aarde, verder weg en ronder, krijgt er maar een
paar. Je kunt deze precessie zien gebeuren in de echte app: schakel naar de
RT-motor (druk op `G`) en draai de sterkte op met `]` om de baan een rozet te zien
trekken.

## Waar dit in de echte app zit

Dit is de gesloten-vormcontrole achter `src/physics/forces.rs`; de werkelijke extra
trekkracht (de "1-post-Newtoniaanse" term) wordt daar bij de versnelling opgeteld en
stap voor stap verkend in de educatieve modus van de app (druk op `K`).

🎉 **Dat is de volledige proof-of-concept-ladder.** Je hebt nu rechtelijnige
beweging, zwaartekracht, twee integratoren, energie, de vergelijking van Kepler en
een relativistische voorspelling geschreven — het echte hart van een
zonnestelselsimulator.
