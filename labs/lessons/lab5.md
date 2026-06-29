# Practicum 5 — Energie, en het viriaaltheorema

**Doel:** de kinetische en potentiële energie van een lichaam berekenen.
**Jij past aan:** `energies` in `src/lib.rs`. **Draai:** `cargo test --test lab5`

---

## Het idee

Er zijn twee soorten energie in een baan:

- **Kinetische** — de bewegingsenergie: `KE = ½·m·v²` (altijd positief).
- **Potentiële** — energie opgeslagen in de zwaartekracht: `PE = −G·M·m / r`. Die is
  **negatief**, omdat je energie zou moeten *toevoegen* om het lichaam van de zon weg
  te slepen (naar `r → ∞`, waar `PE → 0`).

Geef ze terug als paar `(KE, PE)`. In code: `v.length_squared()` is `v²` en
`r.length()` is `r`; `gm_sun` is `G·M`.

## Waarom dit belangrijk is

Voor een gesloten systeem kan de **totale** energie `KE + PE` niet veranderen — die
blijft behouden. Dat is een krachtige ingebouwde controle: als de totale energie van
een simulatie afdwaalt, gaan de getallen de fout in (je zag Euler dit doen in
Practicum 3). De echte app tekent dit live — druk op `Y` voor de energiegrafiek en
kijk hoe de totaallijn vlak blijft.

Er is ook het elegante **viriaaltheorema**. Voor een cirkelbaan voldoet de snelheid
aan `v² = G·M/r`, en als je dat in de twee formules invult, vind je:

```
KE = ½·m·(G·M/r),   PE = −m·(G·M/r)   ⇒   2·KE + PE = 0
```

De kinetische energie is dus precies half zo groot als de (negatieve) potentiële
energie. De test controleert deze gelijkheid — die alleen klopt als *beide* formules
goed zijn.

## Dieper graven

Het volledige verhaal — hoe de bewegingsvergelijkingen zelf uit de energie volgen
(de **Hamiltoniaan**), en een nette afleiding van het viriaaltheorema — staat in
[`../docs/math-nl.md`](../docs/math-nl.md), paragraaf 12.

➡️ **Volgende:** [Practicum 6 — een planeet op zijn ellips vinden (vergelijking van
Kepler)](lab6.md).
