# Practicum 6 — Waar is de planeet? (vergelijking van Kepler)

**Doel:** een vergelijking oplossen die je niet met algebra kunt oplossen.
**Jij past aan:** `solve_kepler` in `src/lib.rs`. **Draai:** `cargo test --test lab6`

---

## Het idee

Een planeet beweegt niet met een vaste hoek rond zijn ellips — hij versnelt dicht
bij de zon en vertraagt ver weg (de tweede wet van Kepler). Astronomen lossen dit op
met twee hoeken:

- de **gemiddelde anomalie** `M` — een nephoek die *wel* gelijkmatig met de tijd
  groeit (als een klokwijzer), en
- de **excentrische anomalie** `E` — de hoek die de plaats van de planeet op de
  ellips echt vastlegt.

Ze zijn aan elkaar verbonden door de **vergelijking van Kepler**:

```
M = E − e·sin E
```

waarbij `e` de excentriciteit is (hoe uitgerekt de ellips is). Het probleem: je
krijgt `M` en `e` en wilt `E` — maar je kunt dit niet voor `E` herschrijven, omdat
`E` zowel binnen als buiten een sinus zit.

## De methode: de methode van Newton

Als je een vergelijking niet rechtstreeks kunt oplossen, ga je *gokken en
verbeteren*. Schrijf de vergelijking als "vind `E` waarvoor `f(E) = E − e·sin E − M =
0`". De methode van Newton springt naar waar de raaklijn de nul raakt:

```
E ← E − f(E) / f′(E)     met   f′(E) = 1 − e·cos E
```

Begin bij `E = M` en herhaal een handvol keer (8 is ruim voldoende). Elke ronde
*verdubbelt* ongeveer het aantal correcte cijfers, dus het convergeert vrijwel
onmiddellijk.

## Wat je moet doen

```rust
let mut big_e = mean_anomaly;
for _ in 0..8 {
    let f  = big_e - eccentricity * big_e.sin() - mean_anomaly;
    let fp = 1.0 - eccentricity * big_e.cos();
    big_e -= f / fp;
}
big_e
```

## Controleer jezelf

```text
cargo test --test lab6
```

Drie tests: bij `e = 0` is het antwoord gewoon `M`; het antwoord voldoet altijd aan
`M = E − e·sin E` met hoge precisie; en het komt overeen met een bekende waarde uit
Meeus' *Astronomical Algorithms* (`M = 5°, e = 0,0167 → E ≈ 5,0855°`). De echte app
gebruikt precies deze oplosser voor de manen (`src/astro/elements.rs`).

➡️ **Volgende (uitdaging):** [Practicum 7 — Mercurius en Einstein](lab7.md).
