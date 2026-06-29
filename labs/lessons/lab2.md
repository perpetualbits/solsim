# Practicum 2 — De zwaartekracht

**Doel:** de versnelling berekenen die de zon aan een lichaam geeft.
**Jij past aan:** `gravity_acceleration` in `src/lib.rs`.
**Jij draait:** `cargo test --test lab2`

---

## Het idee

In Practicum 1 dreef het lichaam in een rechte lijn. Echte planeten krommen, omdat
de zon eraan **trekt**. Newtons gravitatiewet vertelt ons hoe sterk die trekkracht
is en welke kant hij op wijst.

**Sterkte.** De versnelling wordt zwakker met het *kwadraat* van de afstand:

```
|a| = G·M / r²
```

Hier is `G·M` (in de code noemen we het `gm`) één getal dat de zwaartekracht van de
zon meet, en `r` is de afstand tot de zon. Twee keer zo ver weg → nog maar een
kwart van de trekkracht. Dit is de beroemde **kwadratenwet** (omgekeerd evenredig
met het kwadraat van de afstand).

**Richting.** De zwaartekracht wijst recht *naar* de zon, die in de oorsprong
`(0, 0, 0)` staat. De positievector `r⃗` van het lichaam wijst de andere kant op —
*van* de zon *naar* het lichaam — dus we hebben een minteken nodig om hem om te
draaien.

Sterkte en richting samen geven de vectorformule:

```
a⃗ = −G·M · r⃗ / |r⃗|³
```

Waarom de **derde macht** `|r⃗|³` onderin? Twee van die machten zijn de
kwadratensterkte (`/r²`). De derde maakt van de positievector `r⃗` een zuivere
*richting* met lengte 1 (een vector delen door zijn eigen lengte geeft een
eenheidsvector). De formule is dus eigenlijk "sterkte × richting", verstopt in één
nette uitdrukking.

## Wat je moet doen

Vul in `src/lib.rs` de functie `gravity_acceleration` in:

1. Bereken de afstand: `let len = r.length();` (dit is `|r⃗|`).
2. Geef terug: `-gm * r / (len * len * len)`.

(`len * len * len` is `|r⃗|³`. Je mag ook `len.powi(3)` schrijven.)

## Een uitgewerkt getal

Bij `r = 1 AE` met de `gm = GM_SUN` van de zon is de afstand 1, dus `|a| = GM_SUN`
(ongeveer `2,96×10⁻⁴` AE/dag²), en hij wijst in de −x-richting. De test controleert
precies dit, plus de kwadratenafname en de richting in volledig 3D.

## Controleer jezelf

```text
cargo test --test lab2
```

## Denk er eens over na

- Hoe zouden banen eruitzien als de zwaartekracht afnam als `1/r` of `1/r³` in
  plaats van `1/r²`? (Het blijkt dat `1/r²` bijzonder is: het is een van slechts
  twee krachtwetten die banen geven die netjes tot een ellips sluiten. De echte
  simulator laat je dit idee verkennen.)

➡️ **Hierna:** Practicum 3 combineert Practicum 1 en 2 tot een echte tijdstap — en
laat zien waarom de voor de hand liggende manier de baan langzaam laat spiraliseren,
waardoor we later een slimmere methode (RK4) nodig hebben.
