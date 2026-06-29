# Practicum 3 — Eén tijdstap (de simpele manier), en waarom hij afdwaalt

**Doel:** beweging (Practicum 1) en zwaartekracht (Practicum 2) samenvoegen tot één
tijdstap.
**Jij past aan:** `euler_step` in `src/lib.rs`. **Draai:** `cargo test --test lab3`
**Nodig:** Practicum 2 eerst af (deze roept jouw `gravity_acceleration` aan).

---

## Het idee

Een simulatie is eigenlijk: *bereken de trekkracht, verander de snelheid, beweeg,
herhaal.* De eenvoudigste manier om één stap op te schrijven heet **voorwaartse
Euler**. Met de waarden aan het **begin** van de stap:

```
a     = gravity_acceleration(r, gm)   // de trekkracht op dit moment
r_nieuw = r + v·dt                     // beweeg met de OUDE snelheid
v_nieuw = v + a·dt                     // werk daarna de snelheid bij
```

Geef `(r_nieuw, v_nieuw)` terug. Draai dit in een lus en het lichaam beschrijft een
baan.

## De adder onder het gras

Voorwaartse Euler is makkelijk, maar niet erg goed. Omdat hij aanneemt dat de
snelheid en de trekkracht de hele stap lang vastliggen, schiet hij steeds een beetje
**door** — en voor een baan betekent dat dat hij langzaam **energie wint** en naar
buiten spiraliseert. De tweede test controleert dit met opzet: begin op een perfecte
cirkel, doe veel stappen, en de straal groeit. Dat is geen fout in jouw code; het is
de methode zelf die grof is.

Dit is de motivatie voor Practicum 4, waar RK4 het afdwalen oplost door binnen de
stap vooruit te kijken. (Je kunt energieafdwaling ook live *zien* in de echte app:
druk op `Y` voor de energiegrafiek.)

## Denk er eens over na

- Waarom leidt het gebruik van de *oude* snelheid om te bewegen tot doorschieten bij
  een krommende baan?
- Wat gebeurt er met de afdwaling als je `dt` kleiner maakt? (Kleinere stappen
  helpen — maar een betere *methode* helpt veel meer, voor evenveel werk.)

➡️ **Volgende:** [Practicum 4 — een stap die niet afdwaalt (RK4)](lab4.md).
