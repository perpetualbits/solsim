# Practicum 4 — Een tijdstap die niet afdwaalt (RK4)

**Doel:** een nauwkeurige stap nemen zodat banen gesloten blijven.
**Jij past aan:** `rk4_step` in `src/lib.rs`. **Draai:** `cargo test --test lab4`
**Nodig:** Practicum 2 eerst af.

---

## Het idee

Voorwaartse Euler (Practicum 3) vertrouwde op één rechtelijnige schatting voor de
hele stap, en dwaalde af. **Runge–Kutta van de 4e orde (RK4)** doet het veel beter:
hij bemonstert de beweging **vier keer** verspreid over de stap — aan het begin, op
twee tussenpunten en aan het eind — en mengt ze met de gewichten 1-2-2-1, wat het
grootste deel van de fout wegwerkt.

Onze toestand is het paar `(r, v)`, en de veranderingssnelheid ervan is `(v, a(r))`
— snelheid verandert de positie, versnelling verandert de snelheid. De vier
bemonsteringen zijn:

```
k1r = v             k1v = a(r)
k2r = v + ½dt·k1v   k2v = a(r + ½dt·k1r)
k3r = v + ½dt·k2v   k3v = a(r + ½dt·k2r)
k4r = v + dt·k3v    k4v = a(r + dt·k3r)
```

en de nieuwe toestand is het gewogen gemiddelde:

```
r_nieuw = r + dt/6·(k1r + 2·k2r + 2·k3r + k4r)
v_nieuw = v + dt/6·(k1v + 2·k2v + 2·k3v + k4v)
```

waarbij `a(p)` staat voor `gravity_acceleration(p, gm)`.

## Wat je moet doen

Een kleine closure houdt het leesbaar:

```rust
let a = |p: glam::DVec3| gravity_acceleration(p, gm);
```

Bereken daarna `k1…k4` precies zoals hierboven en geef `(r_nieuw, v_nieuw)` terug.

## Controleer jezelf

```text
cargo test --test lab4
```

Twee tests: een cirkelbaan blijft rond tot op een duizendste AE over 400 stappen
(dat lukte Euler niet!), en na één volledige omloop komt het lichaam terug op zijn
startpunt. Dit is precies de methode die de echte `solarsim` gebruikt
(`src/physics/nbody.rs`).

## Denk er eens over na

- Vergelijk met Practicum 3: evenveel *stappen*, maar RK4 doet ~4× zoveel werk per
  stap. Waarom is die ruil meestal de moeite waard? (Hint: Eulers fout halveren
  vraagt veel meer stappen; de fout van RK4 krimpt veel sneller als de stap kleiner
  wordt.)

➡️ **Volgende:** [Practicum 5 — energie, en controleren of die behouden blijft](lab5.md).
