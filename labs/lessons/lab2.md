# Lab 2 — The pull of gravity

**Goal:** compute the acceleration the Sun gives a body.
**You edit:** `gravity_acceleration` in `src/lib.rs`.
**You run:** `cargo test --test lab2`

---

## The idea

In Lab 1 the body coasted in a straight line. Real planets curve, because the Sun
**pulls** on them. Newton's law of universal gravitation tells us how strong that
pull is and which way it points.

**Strength.** The acceleration gets weaker with the *square* of the distance:

```
|a| = G·M / r²
```

Here `G·M` (we call it `gm` in the code) is a single number measuring the Sun's
gravity, and `r` is the distance to the Sun. Twice as far away → only a quarter of
the pull. This is the famous **inverse-square law**.

**Direction.** Gravity points straight *at* the Sun, which sits at the origin
`(0, 0, 0)`. The body's position vector `r⃗` points the other way — *from* the Sun
*to* the body — so we need a minus sign to turn it around.

Putting strength and direction together gives the vector formula:

```
a⃗ = −G·M · r⃗ / |r⃗|³
```

Why the **cube** `|r⃗|³` on the bottom? Two of those powers are the inverse-square
strength (`/r²`). The third one turns the position vector `r⃗` into a pure
*direction* of length 1 (dividing a vector by its own length gives a unit vector).
So the formula is really "strength × direction" hidden in one tidy expression.

## What to do

In `src/lib.rs`, fill in `gravity_acceleration`:

1. Get the distance: `let len = r.length();` (this is `|r⃗|`).
2. Return `-gm * r / (len * len * len)`.

(`len * len * len` is `|r⃗|³`. You could also write `len.powi(3)`.)

## A worked number

At `r = 1 AU` with the Sun's `gm = GM_SUN`, the distance is 1, so `|a| = GM_SUN`
(about `2.96×10⁻⁴` AU/day²) and it points in the −x direction. The test checks
exactly this, plus the inverse-square fall-off and the direction in full 3-D.

## Check yourself

```text
cargo test --test lab2
```

## Think about it

- What would orbits look like if gravity fell off as `1/r` or `1/r³` instead of
  `1/r²`? (It turns out `1/r²` is special: it is one of only two force laws that
  give orbits which close into a neat ellipse. The real simulator lets you explore
  this idea.)

➡️ **Coming next:** Lab 3 will combine Labs 1 and 2 into a real time step — and
show why doing it the obvious way makes the orbit slowly spiral, which is why we
later need a cleverer method (RK4).
