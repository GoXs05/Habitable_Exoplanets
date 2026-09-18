# Habitable Exoplanets

An **experimental, educational tool** for exploring how a machine learning model
decides whether an exoplanet might be habitable.

It is built on planetary and stellar data from the NASA Exoplanet Archive, and
it has two halves:

- a **command-line classifier** — the original pipeline, which trains a
  `HistGradientBoostingClassifier` on the labelled catalogue and predicts on a
  planet you describe;
- an **interactive explorer** — a browser interface where you invent a planet
  with five sliders and watch the model's probability, its per-prediction
  feature attributions, and its decision boundary update live.

The goal is not to publish habitability predictions. It is to make a model's
reasoning *visible* — including the places where the model, the labels and the
data are weaker than a headline accuracy score would suggest. Several of those
weak points are documented below on purpose; finding and understanding them is
the most useful thing this repository has to teach.

---

## Quick start

```bash
pip install -r requirements.txt
```

**Interactive explorer** (recommended):

```bash
python -m uvicorn app.api:app --reload
# then open http://127.0.0.1:8000
```

On first run this trains the interactive model and writes
`models/habitability_gui_model.joblib` (a few seconds). Later runs load that
artifact at startup.

**Command-line classifier:**

```bash
python run.py                 # opens a Tkinter form, prints Habitable / Unhabitable
python run.py -test -report   # trains, evaluates, prints a classification report
```

To retrain the interactive model explicitly:

```bash
python -m src.train_gui_model
```

---

## The interactive explorer

<!-- Everything below is live in the browser; no step requires the CLI. -->

**Five inputs.** Mass, radius, surface temperature and orbital period are
sliders; planet type is a toggle group. Mass, temperature and period are
log-scaled, because their real ranges span four to nine orders of magnitude and
a linear slider would spend almost all its travel in a region containing a
handful of planets. Every slider is bounded to the **1st–99th percentile of the
actual training data**, so you cannot dial in a planet the model has no basis
for.

**A live prediction.** The probability comes from `predict_proba`, shown as a
number and an animated dial. Calls are debounced 175 ms, so dragging a slider
produces one request when you stop, not one per pixel.

**Why it decided that.** Alongside the probability, the interface shows the top
feature contributions for *that specific prediction*, computed with exact
Shapley values over the five inputs (2⁵ = 32 coalitions — exact, not sampled).
They satisfy the additivity property: the contributions sum to
`probability − baseline`. This is the part worth playing with. Set a temperate
Earth-sized world and one feature dominates completely; the section on
[what the model actually learned](#what-the-model-actually-learned) explains why.

**The decision boundary.** A scatter plot puts orbital period on the x-axis and
surface temperature on the y-axis, shaded by the model's predicted probability
across an 80×72 grid of that plane. Real archive planets are plotted as
background points, and your constructed planet as a distinct animated marker.
By default the surface is a slice through your *current* mass, radius and type;
a toggle switches it to hold them at their training medians. This is the fastest
way to see that the model's boundary is a narrow horizontal band.

**A rendered planet.** Your planet is drawn as a procedurally shaded sphere you
can drag to rotate. It transforms as you move the sliders:

| Input | What it changes |
|---|---|
| Radius | The sphere's size |
| Surface temperature | Surface type — frozen, temperate, arid, molten — blended smoothly, with emissive glow above ~800 K |
| Mass | Atmospheric halo thickness |
| Planet type | Terrain: rocky worlds get continents, oceans and polar caps; Jovian and Neptunian get banded turbulence, and Jovian gets rings. Gas giants are drawn oblate |
| Orbital period | The **orbit diagram** (see below) |
| Model probability | A faint biosphere tint and the atmosphere's colour |

The globe's rotation is **not** driven by any input. A planet's day and its year
are independent quantities, and the catalogue has no rotation-period column, so
the spin is a viewing aid only.

**An orbit diagram.** Orbital period drives a separate top-down diagram of the
planet circling its host star. The orbit's size is derived, not decorative:
Kepler's third law gives

```
a = (P / 365.25 days)^(2/3) AU        for a 1 M☉ host
```

so moving the period slider moves the planet in and out exactly as the physics
requires. Checked against the solar system, this reproduces Mercury (0.387 AU),
Venus (0.723), Earth (1.000), Mars (1.524) and Jupiter (5.201) to within 0.05%.
Only the *lap time* is compressed — real periods here span 0.67 to 14,006 days.

---

## The data

| File | Rows | Contents |
|---|---|---|
| `data/all_exoplanets.csv` | 5,599 | The full catalogue, 29 columns |
| `data/habitable_exoplanets.csv` | 70 | The positive class |
| `data/unhabitable_exoplanets.csv` | 5,529 | The negative class |

Each row carries planetary parameters (mass, radius, orbital period,
semi-major axis, gravity, density, flux, equilibrium and surface temperature,
periastron/apastron, Hill sphere), host-star parameters (mass, radius,
temperature, luminosity, magnitude, distance, snow line), and categorical
classifications (planet type, thermal class, stellar spectral class).

**Provenance.** The planetary and stellar measurements trace back to the NASA
Exoplanet Archive. Several other columns — `Planet_Habitability`,
`Planet_Earth_Similarity_Index`, `Planet_Temperature_Type`,
`Planet_Habitability_Zone_Location`, `Star_Snow_Line` — are **not** NASA Archive
fields; they follow the conventions of the Planetary Habitability Laboratory
(PHL) Exoplanet Catalog, which derives them from archive data. If you intend to
cite this work, confirm the exact upstream source and snapshot date first.

**Labels.** `Planet_Habitability` is `0` (unhabitable), or `1`/`2` for two
flavours of habitability, which the pipeline collapses to a single positive
class. That leaves **70 positives against 5,529 negatives — a 1.25% positive
rate.**

That imbalance is the single most important fact about this dataset. A model
that answers "unhabitable" every single time scores **99.5% accuracy**. Accuracy
is therefore meaningless here, and this repository reports precision, recall and
PR-AUC instead.

**Planet types** are `Terran`, `Superterran`, `Subterran`, `Miniterran`,
`Neptunian` and `Jovian`. Only three have any habitable examples at all:

| Type | In data | Labelled habitable |
|---|---|---|
| Jovian | 1,706 | 0 |
| Neptunian | 1,401 | 0 |
| Superterran | 1,347 | 41 |
| Terran | 1,060 | 28 |
| Subterran | 69 | 1 |
| Miniterran | 9 | 0 |

**Missing values matter.** `Planet_Surface_Temperature` is absent for **56.4%**
of rows and is imputed with the column mean during training — which is worth
remembering, since it turns out to be the feature the model leans on hardest.

---

## The two models

Both use the same estimator and preprocessing: mean-imputed numeric features,
most-frequent-imputed and one-hot-encoded categoricals, feeding a
`HistGradientBoostingClassifier` (scikit-learn 1.9.1).

### 1. The full pipeline — `src/prepare_datasets.py`, `src/prep_hist_model.py`

Used by `run.py`. It drops five identifier columns and trains on **everything
else — 24 features**. It reports a perfect score:

```
precision  recall  f1-score
    1.000   1.000     1.000
```

**That number is label leakage, not skill.** Among those 24 features is
`Planet_Earth_Similarity_Index`, which averages **0.25 for unhabitable planets
and 0.74 for habitable ones** — it is essentially a habitability score computed
from the same reasoning that produced the labels. `Planet_Temperature_Type`,
`Planet_Flux` and `Planet_Equilibrium_Temperature` carry similar information.
The model is reading the answer off the input.

A perfect cross-validated score on a 1.25%-positive astronomy dataset is a
result to distrust, and tracking down *why* is a better exercise than the score
itself.

### 2. The interactive model — `src/train_gui_model.py`

Trained on **only the five features the interface exposes**, and saved to
`models/habitability_gui_model.joblib` together with its slider bounds, medians
and metrics.

It exists because the 24-feature model cannot be explored interactively. Sweeping
1,800 combinations of the five user-facing inputs, with the other 19 features
held at their medians, the full model's output spans **0.0000 to 0.0016 — three
distinct values.** The other features dominate so completely that the sliders do
nothing. The five-feature model spans the full 0–1 range and responds to every
input.

Honest cross-validated performance (5-fold stratified, on all 5,599 planets):

| Metric | Value |
|---|---|
| ROC-AUC | 0.999 |
| **PR-AUC** | **0.821** |
| Precision (habitable) | 0.812 |
| Recall (habitable) | 0.800 |
| Accuracy | 0.995 *(see the caveat above)* |

PR-AUC is the number to read. ROC-AUC flatters heavily imbalanced problems;
0.821 against a 0.0125 baseline is a genuinely good result, and far more
informative than the 0.995 accuracy beside it.

### What the model actually learned

Query the API with Earth's values and the attributions come back like this:

```
P(habitable) = 0.9993     baseline 0.0200
  Surface temperature   +0.940
  Radius                +0.016
  Orbital period        +0.015
  Planet type           +0.007
  Mass                  +0.002
```

**Surface temperature accounts for almost the entire prediction.** Sweeping the
2-D grid confirms it: the habitable region is a narrow band at roughly
**233–322 K** — close to the range where liquid water is possible — and it
extends across every orbital period. Planet type barely registers: a Jovian with
Earth's mass, radius and temperature still scores 0.998 against a Terran's 0.999.

This is a defensible thing for the model to have learned, and also a limitation.
It has essentially rediscovered "temperate surface temperature implies
habitable" — which is close to how the labels were defined in the first place.
It has not learned anything about atmospheric composition, magnetic fields,
stellar activity or tidal locking, because none of that is in the data.

---

## API

The backend loads the model once at startup and serves two endpoints.

### `GET /api/meta`

Bootstraps the interface: feature definitions with slider bounds and scales,
planet types with their class counts, training medians, cross-validated metrics,
and the 2,441 archive planets that have both an orbital period and a surface
temperature (used as background points on the plot).

### `POST /api/predict`

```json
{
  "mass": 1.0,
  "radius": 1.0,
  "surface_temperature": 288.0,
  "period": 365.25,
  "planet_type": "Terran",
  "include_surface": false
}
```

Returns the probability, the SHAP baseline, and per-feature contributions sorted
by absolute magnitude:

```json
{
  "probability": 0.9993,
  "base_value": 0.0200,
  "shap": [
    { "feature": "Planet_Surface_Temperature", "label": "Surface temperature",
      "value": 288.0, "contribution": 0.9401, "direction": "up" }
  ],
  "surface": null
}
```

Set `include_surface: true` with `surface_x` and `surface_y` axis descriptors to
also receive the probability grid. An optional `surface_hold` object pins the
three non-plotted features while the grid is swept.

Typical latency: **~18 ms** for a prediction with SHAP, **~50 ms** including the
probability surface — comfortably inside the 175 ms debounce.

---

## Project structure

```
data/                        Three CSVs: full catalogue, positives, negatives
models/                      Saved interactive model (generated)
run.py                       CLI entry point
src/
  prepare_datasets.py        Loads and splits the CSVs for the 24-feature model
  prep_hist_model.py         Builds the sklearn pipeline
  hist_gradient_boosting.py  Fits and classifies a single user-entered planet
  hist_gradient_boosting_tester.py   Train/test evaluation
  user_input.py              Tkinter input form
  train_gui_model.py         Trains and persists the 5-feature interactive model
app/
  api.py                     FastAPI service
  static/
    index.html               Interface markup
    style.css                Styling
    app.js                   Controls, debouncing, plot, state
    planet.js                Procedural planet renderer and orbit diagram
```

Note that `hist_gradient_boosting.classify()` **retrains the pipeline on every
call** — it does not persist a model. Only the interactive model is saved.

---

## Limitations

Read this section before drawing any conclusion from the output.

1. **70 positive examples.** Every performance figure rests on them. A handful of
   relabelled planets would move the metrics noticeably.
2. **The labels are derived, not observed.** No planet in this catalogue is
   *known* to be habitable. The labels encode a model of habitability, so the
   classifier is learning to reproduce someone else's heuristic.
3. **The CLI model leaks.** Its perfect score is an artifact. Do not quote it.
4. **The interactive model is effectively a temperature threshold**, and 56.4%
   of surface temperatures are imputed.
5. **Trees do not extrapolate.** Push a slider past the training range and the
   prediction flattens rather than extending sensibly.
6. **Habitability is far more than five numbers.** Atmosphere, magnetic field,
   stellar activity, tidal locking and system architecture all matter and none
   are represented here.

This is a learning project. It is a good way to build intuition about class
imbalance, label leakage, feature attribution and decision boundaries — and a
poor way to decide where to point a telescope.

---

## Background: the science

**What is an exoplanet?** Any planet outside our Solar System. Thousands have
been found since the 1990s, largely by missions such as Kepler and TESS.

**The habitable zone.** The orbital region around a star where a planet's
surface temperature could permit liquid water — sometimes called the Goldilocks
zone. Its distance depends on the star's size, temperature and luminosity, so it
sits much closer in around a cool M dwarf than around a hot F star.

**Why these features?** Mass and radius together give surface gravity and bulk
density, which govern whether a world can hold an atmosphere. Orbital period
fixes the orbit's size (Kepler's third law) and therefore how much stellar
energy the planet receives. Surface temperature is the most direct indicator of
whether liquid water can persist. Planet type is a size-and-mass classification:
Terran worlds are broadly Earth-like, Superterran are larger rocky planets
(often called super-Earths), while Neptunian and Jovian worlds are gas-dominated
and not candidates for surface life.

For reference, Earth: mass 1 M⊕, radius 1 R⊕, orbital period 365.25 days,
mean surface temperature 288 K, type Terran. Use the **Earth** button in the
interface to snap every input to those values.

---

## License

MIT — see [LICENSE](LICENSE).
