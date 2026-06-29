# Practicum 1 — Laat een hemellichaam in een rechte lijn bewegen

**Doel:** een lichaam met een constante snelheid door de ruimte laten zweven.
**Jij past aan:** `advance_position` in `src/lib.rs`.
**Jij draait:** `cargo test --test lab1`

---

## Het idee

Voordat we zwaartekracht toevoegen, doen we de eenvoudigst mogelijke beweging: een
lichaam dat met een constante snelheid voortdrijft, zonder dat er iets aan duwt.

Deze regel ken je al uit de natuurkunde:

> afstand = snelheid × tijd

Een simulator doet precies dat, maar in kleine stapjes en in 3D. We houden twee
dingen van het lichaam bij:

- de **positie** `r` — waar het is (een punt in de ruimte), en
- de **snelheid** `v` — hoe snel en in welke richting het beweegt.

Na een kleine tijdstap `dt` is het lichaam `v·dt` verder bewogen (snelheid × tijd,
maar als vector zodat de richting behouden blijft). De nieuwe positie is dus:

```
r_nieuw = r + v·dt
```

Dat is het hele practicum. Het lijkt bijna te simpel — maar deze ene regel keer op
keer herhalen, terwijl de zwaartekracht langzaam `v` verandert, is wat planeten
laat draaien. Je bouwt het **geraamte** van de simulator.

## Waarom vectoren?

`r` en `v` zijn van het type `DVec3`: elk bevat drie getallen (x, y, z). Door
`r + v * dt` te schrijven tel je de stap bij alle drie tegelijk op, zodat het
lichaam in elke richting goed beweegt, niet alleen langs één as. Met glam kun je een
vector vermenigvuldigen met een getal (`v * dt`) en twee vectoren optellen
(`r + ...`).

## Wat je moet doen

Open `src/lib.rs`, zoek `advance_position` en vervang de `todo!(...)` (en de regel
`let _ = ...` erboven) door de formule. Het is één regel.

## Controleer jezelf

```text
cargo test --test lab1
```

Je zou drie groene tests moeten zien: één stap beweegt met `v·dt`, veel stappen
tellen op, en het werkt in 3D. Rood? Lees de melding — die vertelt je wat er
verwacht werd tegenover wat jouw code teruggaf.

## Denk er eens over na

- Als `dt` heel groot is, is "een rechte lijn voor de hele stap" dan nog een goede
  schatting zodra de zwaartekracht de baan kromt? (Houd deze vraag in gedachten —
  hij komt terug in Practicum 3, waar we een naïeve methode zien afdwalen.)

➡️ **Volgende:** [Practicum 2 — de zwaartekracht](lab2.md), waar `v` eindelijk gaat
veranderen.
