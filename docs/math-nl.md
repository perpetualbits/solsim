# De wiskunde & natuurkunde van de zonnestelselsimulator

*Dit is de Nederlandse vertaling van [`math-en.md`](math-en.md).*

Dit is de complete naslag voor elke formule, elk algoritme en elk natuurkundig
principe dat het programma gebruikt, met verwijzingen naar het bronbestand waar elk
ding leeft. Het is geschreven voor een **4 VWO**-lezer (≈ 15–16 jaar): eenvoudige
wiskunde in Unicode, elk symbool uitgelegd. De ingebouwde handleiding (druk op
**F1**) is de korte versie van dit document.

## Eenheden en constanten

Alles wat astronomisch is, wordt berekend in **f64** (dubbele precisie); waarden
worden pas naar **f32** omgezet op de grens met de GPU, met een om de camera
gecentreerde *zwevende oorsprong* zodat piepkleine details (op maanschaal) bewaard
blijven.

| Grootheid | Symbool | Waarde | Waar |
|---|---|---|---|
| Lengte | AE | astronomische eenheid (gemiddelde afstand aarde–zon) | — |
| Tijd | dag | — | — |
| Hoek | rad | radialen | — |
| Gaussische gravitatieconstante | k | 0.017202098950 | `astro/constants.rs` |
| Gravitatieparameter van de zon | GM_zon = k² | 2.9591×10⁻⁴ AE³/dag² | `astro/constants.rs` |
| Lichtsnelheid | c | 173.144 AE/dag | `astro/constants.rs` |
| Scheefstand van de ecliptica | ε | 23.4393° | `astro/constants.rs` |
| 1 AE in km | — | 149 597 870.7 | `astro/constants.rs` |

De gravitatieparameter van elk hemellichaam is `G·m = GM_zon · (m / M_zon)`
(`bodies.rs`, `PLANET_GM`).

## 1. Tijd: Juliaanse datums — `astro/time.rs`

Een **Juliaanse datum (JD)** is het aantal dagen sinds een vast nulpunt. De
referentie **J2000** = 2000-01-01 12:00 is `JD = 2451545.0`.

- Kalender → JD gebruikt het algoritme van Meeus (via de `astro`-crate). We bouwen
  de decimale dag zelf op als `dag + (uur + min/60 + sec/3600)/24` (de eigen helper
  van de crate gaat fout met minuten/seconden) en roepen daarna `julian_day` aan.
- De systeemklok (seconden sinds 1970-01-01) wordt een JD met
  `JD = 2440587.5 + seconden/86400`.
- De klok loopt vooruit met `Δt_dagen = realtime_seconden × speed_factor / 86400`.

**Controle:** `jd_from_calendar(2000,1,1,12,0,0) = 2451545.0` (unittest).

## 2. Efemeriden — `astro/ephemeris.rs`

Een **efemeride** geeft de positie van een lichaam voor elke datum, zonder
stap-voor-stap-simulatie.

- **Planeten — VSOP87.** De `vsop87a`-reeksen tellen veel periodieke termen op van
  de vorm `A·cos(B + C·T)`, met `T = (JD − 2451545)/365250` (Juliaanse millennia),
  wat heliocentrische **ecliptische rechthoekige** coördinaten (AE) geeft in het
  vaste J2000-stelsel. Elke term is één kleine periodieke verschuiving die ontstaat
  doordat de planeten aan elkaar trekken. `planet_position(planet, jd)` kiest de
  juiste reeks.
- **Maan — ELP.** De maantheorie van `astro` geeft de geocentrische ecliptische
  lengte λ, breedte β en afstand r (km) van de maan. We zetten dat om naar
  rechthoekige coördinaten `(r·cosβ·cosλ, r·cosβ·sinλ, r·sinβ)`, km → AE, en tellen
  de positie van de aarde erbij op.
- **Snelheid via eindige differentie.** `velocity_fd(f, jd, δ) = (f(jd+δ) − f(jd−δ))
  / (2δ)` — een centrale differentie, gebruikt om de fysicamotor te starten.

**Controles:** aarde–zon ≈ 1 AE; aarde–maan ≈ 0,0024–0,0027 AE (unittests).

## 3. Coördinatenstelsels — `astro/constants.rs`, `render/viewpoints.rs`, `stars/project.rs`

We werken in het **ecliptische J2000**-stelsel (het baanvlak van de aarde, zon in
de oorsprong). Sterrencatalogi gebruiken het **equatoriale** stelsel, dat alleen
verschilt door de astilt ε van de aarde. Eén rotatie om de gedeelde x-as zet
equatoriaal → ecliptisch om:

```
x' = x
y' = y·cosε + z·sinε
z' = −y·sinε + z·cosε
```

De equatoriale eenheidsvector uit rechte klimming α en declinatie δ is
`(cosδ·cosα, cosδ·sinα, sinδ)` (`stars/project.rs`).

## 4. Kepler-banen voor de manen — `astro/elements.rs`

De meeste manen gebruiken gemiddelde **Kepler-elementen** `(a, e, i, Ω, ω, M₀,
periode)`:

1. De gemiddelde anomalie groeit lineair: `M = M₀ + 2π·(JD − epoche)/periode`.
2. Los **de vergelijking van Kepler** `M = E − e·sin E` op voor de excentrische
   anomalie `E` met Newton–Raphson: `E ← E − (E − e·sinE − M)/(1 − e·cosE)`.
3. Ware anomalie `ν = 2·atan2(√(1+e)·sin(E/2), √(1−e)·cos(E/2))`, straal
   `r = a·(1 − e·cosE)`; punt in het vlak `(r·cosν, r·sinν, 0)`.
4. Draai met ω (argument van het periapsis), i (inclinatie) en Ω (knoop) naar het
   ecliptische stelsel, en tel daarna de positie van de moederplaneet erbij op.

Dit is de **eerste wet van Kepler** (een ellips met de planeet in één brandpunt) en
de **tweede wet** (sneller dicht bij het periapsis). De posities zijn benaderde
gemiddelde waarden.

**Controles:** de opgeloste `E` voldoet aan de vergelijking van Kepler; een
cirkelbaan houdt een constante straal (unittests).

## 5. De fysicamotor — `physics/forces.rs`, `physics/nbody.rs`

Als de motor Newtoniaans of Relativistisch is, worden de posities **berekend** door
de bewegingsvergelijkingen te integreren (de zon staat vast in de oorsprong; de acht
planeten worden geïntegreerd; manen rijden analytisch mee op hun geïntegreerde
moeder).

### Newtoniaanse zwaartekracht

Versnelling van lichaam *i* (`forces::accelerations`):

```
a_i = −GM_zon·r_i/|r_i|³  +  Σ_{j≠i} G·m_j·(r_j − r_i)/|r_j − r_i|³
```

De eerste term is de zon; de som is de andere planeten. Dit is **Newtons
gravitatiewet**, `a = G·m / r²`, geschreven als vector.

### Correctie uit de algemene relativiteitstheorie (1PN, zon-gedomineerd)

Tel bij elk lichaam op (r, v ten opzichte van de zon, μ = GM_zon):

```
a_GR = gr_strength · (μ / (c²·r³)) · [ (4μ/r − v²)·r⃗ + 4·(r⃗·v⃗)·v⃗ ]
```

Deze piepkleine extra trekkracht laat de baan langzaam draaien (precessie). De
verschuiving per baan in gesloten vorm is

```
Δϖ = 6π·μ / (c²·a·(1 − e²))
```

Voor **Mercurius** (a = 0,387099 AE, e = 0,205630, periode 87,969 d) is dit
**≈ 43″ per eeuw** — de klassieke toets van de algemene relativiteitstheorie
(`forces::perihelion_advance_arcsec_per_century`). `gr_strength` (toetsen `[` `]`)
overdrijft het voor een snelle visuele indruk; houd het op 1 voor de eerlijke
waarde.

**Controles:** de gesloten vorm geeft 43″/eeuw; bij het integreren van één
Mercuriusbaan verschuift de GR-term het perihelium met het voorspelde bedrag,
terwijl Newtoniaans een gesloten ellips houdt (unittests in `physics/nbody.rs`).

### RK4-integrator — `physics/nbody.rs`

De toestand is `y = (posities, snelheden)`; de veranderingssnelheid ervan is
`(snelheden, versnellingen)`. **Runge–Kutta van de 4e orde** bemonstert die snelheid
vier keer per stap (begin, twee tussenpunten, eind) en combineert ze 1-2-2-1:

```
y_{n+1} = y_n + (dt/6)·(k1 + 2k2 + 2k3 + k4)
```

Dit heft de fout van lage orde op, zodat banen nauwkeurig blijven bij grote stappen.
De tijdstap van elk beeld wordt **onderverdeeld** (deelstappen van ≈ 0,5 dag) voor
nauwkeurigheid; de analytische motor wordt ook onderverdeeld, zodat snelle binnenste
planeten vloeiende banen tekenen in plaats van hoekige koorden (temporele aliasing).
Zie `main.rs` (`PHYSICS_STEP_DAYS`, `TRAIL_STEP_DAYS`). De *grootte* van de stap is
begrensd op `PHYSICS_STEP_DAYS` (`nbody::plan_substeps`), zodat banen met een korte
omlooptijd niet uit elkaar vallen, hoe snel de tijd ook wordt gevraagd; bij extreme
snelheid blijft de tijd dan achter (met een HUD-melding) in plaats van de stap te
vergroven.

## 6. Sterren — `stars/color.rs`, `stars/project.rs`

- **Kleurindex → temperatuur** (Ballesteros 2012):
  `T = 4600·(1/(0.92·(B−V)+1.7) + 1/(0.92·(B−V)+0.62))` K. Blauwe sterren (B−V ≈ 0)
  ≈ 10000 K; rode sterren (B−V ≈ 1,5) ≈ 3800 K.
- **Temperatuur → RGB**: de zwarte-stralerfit van Tanner Helland (heet → blauwwit,
  koel → oranjerood).
- **Magnitude → grootte**: magnitude is logaritmisch en omgekeerd (kleiner =
  helderder), `m₁ − m₂ = −2.5·log₁₀(F₁/F₂)`, dus de getekende stipgrootte is
  `grootte = basis·10^(0.2·(m_ref − m))`, begrensd.
- **Plaatsing**: (α, δ) → equatoriale eenheidsvector → draai met ε → ecliptische
  richting, geschilderd op een verre achtergrondbol die de *rotatie* van de camera
  volgt maar niet de *positie* (`render/starfield.rs`).

De catalogus bevat de 1500 helderste sterren uit de Yale Bright Star Catalogue
(`assets/bsc5.csv`, geladen door `stars/catalog.rs`).

## 7. Camera's en gezichtspunten — `render/camera.rs`, `render/viewpoints.rs`

- **Baancamera.** Positie = `doel + r·(cosφ·cosθ, cosφ·sinθ, sinφ)`; een
  rechtshandige kijkrichting naar het doel met een perspectiefprojectie. De
  voorste/achterste clipvlakken schalen mee met de zoom, zodat je kunt inzoomen van
  het oppervlak van een maan tot het hele stelsel. Alles in f64 gebouwd, naar f32
  omgezet met het doel als zwevende oorsprong.
- **Ecliptisch-noord**: een baancamera gecentreerd op de zon, beginnend bijna recht
  naar beneden.
- **Aardoppervlak**: de lokale **omhoog/noord/oost** van de waarnemer komen uit de
  **lokale sterrentijd** `LST = GMST(jd) + oosterlengte`; het zenit wijst naar
  RK = LST, Dec = breedtegraad in het equatoriale stelsel, gedraaid naar het
  ecliptische. Terwijl de tijd loopt, verandert LST en draait de hemel.
  Standaardwaarnemer: Zutphen (52,14°N, 6,20°O).

## 8. Het referentierooster — `render/grid.rs`

Een 3D-rooster van kubussen rond het brandpunt. Elke lijn **vervaagt met de afstand**
en stopt voorbij een grens. De afstand past zich aan de zoom aan in octaafstappen:
het niveau het dichtst bij `view_scale / IDEAL_CELLS` is het helderst, met een
tweemaal fijner en een tweemaal grover (dikker) niveau die overvloeien met
`frac(log₂ zoom)`. Lijnen worden getekend als rechthoeken in schermruimte (GPU-lijnen
zijn 1 px) zodat de dikte kan variëren. Het is alleen een liniaal — het beïnvloedt de
natuurkunde nooit.

## 9. Logaritmische modus — `render/logscale.rs`

Een vervorming **alleen voor de weergave**. Elk lichaam houdt zijn richting vanaf de
zon, maar zijn afstand `r` wordt vervangen door

```
r_disp = R₀·ln(1 + r/r₀)
```

Omdat `ln` steeds langzamer groeit, worden de ver uit elkaar liggende buitenste
planeten in beeld getrokken zonder dat de binnenste op de zon samenvallen. Toegepast
op lichamen en sporen tijdens het tekenen; de opgeslagen posities en de natuurkunde
veranderen nooit. (Groottes worden vergroot en begrensd zodat lichamen zichtbaar
blijven op de samengeperste schaal; het rooster is verborgen in deze modus.)

---

## 10. Schakelaar voor ware schaal — `bodies::real_radius_au`

Een grooteschakelaar **alleen voor de weergave** (toets `S`). Normaal worden lichamen
veel groter dan in het echt getekend zodat ze zichtbaar zijn. Ware schaal tekent elk
lichaam in plaats daarvan op zijn echte straal:

```
straal_AE = straal_km / 149 597 870.7
```

(149 597 870.7 km = 1 AE.) Dit maakt duidelijk hoe piepklein de lichamen zijn naast
de afstanden ertussen — bijv. de zon ≈ 0,00465 AE, de aarde ≈ 0,0000426 AE. Posities
en natuurkunde blijven ongemoeid; alleen de getekende stralen (en de straal van de
ring van Saturnus) veranderen.

## 11. Educatieve modus — `edu.rs`

Een op zichzelf staande **tweelichamendemo** (toets `K`) die één integratiestap
verbeeldt. De zon staat in de oorsprong; één planeet begint op `r = 1.2 AE` met 0,85×
de cirkelsnelheid `v_circ = √(GM_zon/r)`, wat een duidelijke ellips geeft. Elke stap
gebruikt **semi-impliciete Euler** (met opzet eenvoudiger en groter dan de live RK4):

```
a = −GM_zon·r/|r|³  +  gr_strength·(μ/(c²|r|³))·[(4μ/|r| − v²)·r + 4(r·v)·v]
v ← v + a·Δt
r ← r + v·Δt
```

De stap is opgesplitst in vijf uitlegfasen, die elk meer vectorpijlen tonen: positie
`r`, snelheid getekend als `v·Δt`, het effect van de zwaartekracht als `a·Δt²`, de
GR-correctie (overdreven met `GR_ARROW_EXAG` en begrensd zodat hij zichtbaar is), en
de bijgewerkte snelheid. Snelheidsvectoren worden geschaald met `Δt` en
versnellingsvectoren met `Δt²`, zodat alle pijlen vergelijkbaar zijn als
*verplaatsing per stap* in AE. GR aanzetten laat de baan langzaam precesseren. Dit is
puur een lesweergave; de live simulatie blijft op de achtergrond doortikken maar wordt
niet getoond.

---

## 12. Energie, de Hamiltoniaan en het viriaaltheorema — `physics/energy.rs`

Deze paragraaf legt de energiegrafiek uit (toets `Y`) en, belangrijker, *waarom* de
vergelijkingen die de simulator integreert de juiste zijn. Het is het diepste deel
van de handleiding; neem er de tijd voor.

### 12.1 De twee energieën

Een bewegend lichaam draagt **kinetische energie**, de bewegingsenergie:

```
T = ½·m·v²          (v = |v⃗| is de snelheid)
```

De zwaartekracht slaat **potentiële energie** op. Voor één planeet met massa `m` op
afstand `r` van de zon (massa `M`) is die

```
U = −G·M·m / r
```

`U` is *negatief* en groeit naar 0 als `r → ∞`: dat is gewoon boekhouding voor "je
moet energie *toevoegen* om de planeet weg te slepen tegen de aantrekkingskracht van
de zon in." Voor het hele systeem tellen we de aantrekkingskracht van de zon op elke
planeet op **én** elke onderlinge aantrekkingskracht tussen planeten:

```
T = Σᵢ ½·mᵢ·vᵢ²
U = −Σᵢ G·M·mᵢ/rᵢ  −  Σ_{i<j} G·mᵢ·mⱼ / r_ij
E = T + U                         (de totale energie)
```

In onze eenheden wordt massa gemeten in **zonsmassa's**, dus `M = 1` en
`G = k² = GM_zon`. Dat maakt de code netjes: `G·M·mᵢ = gmᵢ` en
`G·mᵢ·mⱼ = gmᵢ·gmⱼ / GM_zon`, wat precies is wat `system_energy` berekent. (Eenheid
van energie hier: M_zon·AE²·dag⁻².)

### 12.2 Waarom het totaal vlak blijft: behoud van energie

Differentieer `E` naar de tijd en gebruik Newtons wet `mᵢ·a⃗ᵢ = F⃗ᵢ` (de kracht is
`F⃗ᵢ = −∂U/∂r⃗ᵢ`):

```
dE/dt = Σᵢ mᵢ·v⃗ᵢ·a⃗ᵢ + Σᵢ (∂U/∂r⃗ᵢ)·v⃗ᵢ
      = Σᵢ v⃗ᵢ·F⃗ᵢ − Σᵢ v⃗ᵢ·F⃗ᵢ = 0
```

Dus voor pure Newtoniaanse zwaartekracht **kan de totale energie niet veranderen**.
Dat is de witte lijn op de grafiek: hij hoort perfect vlak te zijn. In de praktijk
dwaalt hij een beetje af, omdat RK4 met een eindige stap maar *bij benadering* exact
is, en omdat de **overdreven GR-term een snelheidsafhankelijke kracht is die niet de
gradiënt van een eenvoudige potentiaal is** — met relativiteit aan wordt energie dus
helemaal niet geacht behouden te blijven. De lijn in de gaten houden is daarom een
live controle op hoe betrouwbaar de integratie is.

### 12.3 De Hamiltoniaan — en de beweging *uit* de energie halen

Schrijf de energie met **impuls** `p⃗ᵢ = mᵢ·v⃗ᵢ` in plaats van snelheid. Het
resultaat, gezien als functie van posities `r⃗ᵢ` en impulsen `p⃗ᵢ`, is de
**Hamiltoniaan**:

```
H(r⃗, p⃗) = Σᵢ |p⃗ᵢ|² / (2·mᵢ)  +  U(r⃗)        (= T + U = E)
```

De hele beweging volgt uit `H` via de **vergelijkingen van Hamilton**:

```
dr⃗ᵢ/dt =  ∂H/∂p⃗ᵢ          ṗ⃗ᵢ = −∂H/∂r⃗ᵢ
```

Laten we controleren dat deze *Newton teruggeven*. De eerste:

```
dr⃗ᵢ/dt = ∂H/∂p⃗ᵢ = p⃗ᵢ/mᵢ = v⃗ᵢ          ✓ (impuls is massa × snelheid)
```

De tweede, met `∂U/∂r⃗ᵢ` voor de zonterm `U = −G·M·mᵢ/rᵢ`:

```
ṗ⃗ᵢ = −∂U/∂r⃗ᵢ = −G·M·mᵢ·r⃗ᵢ / rᵢ³
```

Deel door `mᵢ` (omdat `p⃗ᵢ = mᵢ·v⃗ᵢ`, dus `ṗ⃗ᵢ/mᵢ = a⃗ᵢ`):

```
a⃗ᵢ = −G·M·r⃗ᵢ / rᵢ³
```

— precies de versnelling in `physics/forces.rs`. De simulator is dus eigenlijk gewoon
*de vergelijkingen van Hamilton aan het integreren*: de energie `H` **is** de
bewegingswet, en de zwaartekrachtformule die we coderen is wat je krijgt door haar te
differentiëren. Dit is de moderne, verenigende manier om de mechanica te zien; de
GR-correctie is een extra term die erbovenop ligt.

### 12.4 Het viriaaltheorema

Voor de zwaartekracht (een kwadratenkracht, waarbij `U ∝ 1/r`) bestaat er een mooie
relatie tussen de tijdgemiddelde energieën van een gebonden baan:

```
2·⟨T⟩ + ⟨U⟩ = 0          ⇒  ⟨T⟩ = −½·⟨U⟩,  E = ⟨T⟩ + ⟨U⟩ = ½·⟨U⟩ = −⟨T⟩
```

(⟨·⟩ betekent "gemiddeld over één baan".) Het komt voort uit het bekijken van de
grootheid `G = Σ p⃗ᵢ·r⃗ᵢ`: over een volledige, zich herhalende baan is de gemiddelde
veranderingssnelheid ervan nul, en dat uitwerken wordt `2⟨T⟩ + ⟨U⟩ = 0` juist omdat
de kracht gaat als `1/r²`. Een snelle manier om het te vertrouwen: een **cirkelbaan**
voldoet eraan op *elk* moment, niet alleen gemiddeld. Daar is `v² = G·M/r`, dus

```
T = ½·m·v² = ½·G·M·m/r,   U = −G·M·m/r   ⇒   2T + U = 0   ✓
```

Dat cirkelgeval is de unittest in `physics/energy.rs`. Het theorema verklaart de
*vorm* van de grafiek: de totale `E` ligt onder nul, ruwweg halverwege naar de
(negatieve) potentiaallijn, en de kinetische lijn spiegelt die erboven. Het heeft ook
een beroemde astrofysische toepassing — het wegen van sterrenhopen en sterrenstelsels
aan de hand van hoe snel hun leden bewegen — maar hier is het vooral een controle op
de getallen.

---

### Overzicht van bronbestanden

| Onderwerp | Bestand |
|---|---|
| Constanten | `astro/constants.rs` |
| Juliaanse datums, klok | `astro/time.rs` |
| VSOP87-planeten, ELP-maan, snelheid | `astro/ephemeris.rs` |
| Kepler-elementen & -vergelijking | `astro/elements.rs` |
| Catalogus van lichamen & samenstellen | `bodies.rs` |
| Newtoniaanse + GR-krachten | `physics/forces.rs` |
| RK4-integrator, toestand | `physics/nbody.rs` |
| Energie, Hamiltoniaan, viriaal | `physics/energy.rs`, `ui/energy.rs` |
| Sterrenkleuren & -groottes | `stars/color.rs` |
| Plaatsing van sterren | `stars/project.rs`, `render/starfield.rs` |
| Baancamera | `render/camera.rs` |
| Gezichtspunten, sterrentijd | `render/viewpoints.rs` |
| Referentierooster | `render/grid.rs` |
| Logaritmische transformatie | `render/logscale.rs` |
| Stralen op ware schaal | `bodies.rs` (`real_radius_au`) |
| Educatieve modus & vectorpijlen | `edu.rs`, `render/arrows.rs` |
| Sporen | `render/trails.rs` |
| Ingebouwde handleiding | `ui/manual.rs` |
