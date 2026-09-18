"""FastAPI service backing the interactive habitability explorer.

Loads the persisted 5-feature model once at startup (training it first if the
artifact is missing), then serves live predict_proba + per-prediction SHAP
values and a 2D probability surface for the plot.

Run:  python -m uvicorn app.api:app --reload
Then: http://127.0.0.1:8000
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / 'models' / 'habitability_gui_model.joblib'
STATIC_DIR = pathlib.Path(__file__).resolve().parent / 'static'

# SHAP's Exact explainer evaluates 2**5 = 32 feature coalitions against a
# background sample. 100 rows keeps a single explanation at ~10ms.
SHAP_BACKGROUND_SIZE = 100

# Resolution of the probability surface. The decision boundary is a narrow
# temperature band, so it needs enough rows not to smear under interpolation.
GRID_W, GRID_H = 80, 72


class Bundle:
    """Everything loaded once at startup and reused across requests."""

    def __init__(self, payload: dict):
        self.model = payload['model']
        self.feature_cols: list[str] = payload['feature_cols']
        self.numeric_cols: list[str] = payload['numeric_cols']
        self.features: list[dict] = payload['features']
        self.planet_types: list[str] = payload['planet_types']
        self.medians: dict[str, float] = payload['medians']
        self.default_planet_type: str = payload['default_planet_type']
        self.metrics: dict = payload['metrics']
        self.scatter: list[dict] = payload['scatter']
        self.planet_type_counts: dict = payload['planet_type_counts']
        self.habitable_by_type: dict = payload['habitable_by_type']

        self._type_index = {t: i for i, t in enumerate(self.planet_types)}
        # SHAP hands the wrapped function a float matrix, so Planet_Type travels
        # as an index into self.planet_types and is decoded in _to_frame.
        self._lock = threading.Lock()
        self._explainer = self._build_explainer()

    # -- model plumbing ---------------------------------------------------
    def _to_frame(self, arr: np.ndarray) -> pd.DataFrame:
        arr = np.atleast_2d(np.asarray(arr, dtype=float))
        frame = pd.DataFrame(arr, columns=self.feature_cols)
        idx = np.rint(frame['Planet_Type'].to_numpy()).astype(int)
        idx = np.clip(idx, 0, len(self.planet_types) - 1)
        frame['Planet_Type'] = [self.planet_types[i] for i in idx]
        for col in self.numeric_cols:
            frame[col] = frame[col].astype(float)
        return frame[self.feature_cols]

    def _predict_encoded(self, arr: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._to_frame(arr))[:, 1]

    def _build_explainer(self):
        background = self._background_matrix()
        masker = shap.maskers.Independent(background)
        explainer = shap.explainers.Exact(self._predict_encoded, masker)
        # First call carries one-off setup cost (~6s); pay it at startup so the
        # first user interaction is fast.
        explainer(self._encode_row(self.default_input()))
        return explainer

    def _background_matrix(self) -> np.ndarray:
        """Training rows, Planet_Type index-encoded, used as the SHAP baseline."""
        habitable = pd.read_csv(ROOT / 'data' / 'habitable_exoplanets.csv')
        unhabitable = pd.read_csv(ROOT / 'data' / 'unhabitable_exoplanets.csv')
        df = pd.concat([habitable, unhabitable], ignore_index=True)
        frame = df[self.feature_cols].copy()
        frame['Planet_Type'] = (frame['Planet_Type'].map(self._type_index)
                                .fillna(self._type_index[self.default_planet_type]))
        frame = frame.astype(float)
        frame = frame.fillna(frame.median())
        return shap.utils.sample(frame.to_numpy(), SHAP_BACKGROUND_SIZE,
                                 random_state=0)

    def _encode_row(self, values: dict) -> np.ndarray:
        row = [float(values[c]) for c in self.numeric_cols]
        row.append(float(self._type_index.get(values['Planet_Type'], 0)))
        return np.array([row], dtype=float)

    def default_input(self) -> dict:
        out = {c: self.medians[c] for c in self.numeric_cols}
        out['Planet_Type'] = self.default_planet_type
        return out

    # -- public operations ------------------------------------------------
    def predict(self, values: dict) -> float:
        frame = pd.DataFrame([values])[self.feature_cols]
        return float(self.model.predict_proba(frame)[0, 1])

    def explain(self, values: dict) -> tuple[float, list[float]]:
        encoded = self._encode_row(values)
        # shap.Explainer objects are not documented as thread-safe and uvicorn
        # runs sync endpoints in a threadpool, so serialise access.
        with self._lock:
            result = self._explainer(encoded)
        return float(result.base_values[0]), [float(v) for v in result.values[0]]

    def surface(self, held: dict, x_axis: dict, y_axis: dict) -> dict:
        """Probability across the period/temperature plane, other features held."""
        xs = _axis_values(x_axis, GRID_W)
        ys = _axis_values(y_axis, GRID_H)
        xx, yy = np.meshgrid(xs, ys)
        n = xx.size
        grid = pd.DataFrame({
            'Planet_Mass': np.full(n, held['Planet_Mass'], dtype=float),
            'Planet_Radius': np.full(n, held['Planet_Radius'], dtype=float),
            'Planet_Surface_Temperature': yy.ravel().astype(float),
            'Planet_Period': xx.ravel().astype(float),
            'Planet_Type': [held['Planet_Type']] * n,
        })[self.feature_cols]
        proba = self.model.predict_proba(grid)[:, 1]
        return {
            'width': GRID_W,
            'height': GRID_H,
            'x_values': [float(v) for v in xs],
            'y_values': [float(v) for v in ys],
            'values': [round(float(v), 5) for v in proba],
        }


def _axis_values(axis: dict, count: int) -> np.ndarray:
    lo, hi = float(axis['min']), float(axis['max'])
    if axis.get('scale') == 'log':
        return np.logspace(np.log10(lo), np.log10(hi), count)
    return np.linspace(lo, hi, count)


bundle: Bundle | None = None


def _load_bundle() -> Bundle:
    if not ARTIFACT.exists():
        print(f'{ARTIFACT.name} not found -- training it now...', flush=True)
        subprocess.run([sys.executable, '-m', 'src.train_gui_model'],
                       cwd=ROOT, check=True)
    return Bundle(joblib.load(ARTIFACT))


@asynccontextmanager
async def lifespan(_: FastAPI):
    global bundle
    bundle = _load_bundle()
    print(f'Model ready: {len(bundle.scatter)} scatter points, '
          f'PR-AUC {bundle.metrics["pr_auc"]:.3f}', flush=True)
    yield


app = FastAPI(title='Exoplanet Habitability Explorer', lifespan=lifespan)


class SurfaceHold(BaseModel):
    mass: float
    radius: float
    planet_type: str


class PredictRequest(BaseModel):
    mass: float = Field(..., description='Planet mass in Earth masses')
    radius: float = Field(..., description='Planet radius in Earth radii')
    surface_temperature: float = Field(..., description='Surface temperature in K')
    period: float = Field(..., gt=0, description='Orbital period in days')
    planet_type: str
    include_surface: bool = False
    surface_x: dict | None = None
    surface_y: dict | None = None
    # Values to hold the non-plotted features at while sweeping the surface.
    # Omitted -> the surface is a slice through the caller's own planet.
    surface_hold: SurfaceHold | None = None


class ShapContribution(BaseModel):
    feature: str
    label: str
    value: float
    contribution: float
    direction: Literal['up', 'down']


class PredictResponse(BaseModel):
    probability: float
    base_value: float
    shap: list[ShapContribution]
    surface: dict | None = None


def _require_bundle() -> Bundle:
    if bundle is None:
        raise HTTPException(503, 'Model is still loading')
    return bundle


@app.get('/api/meta')
def meta():
    b = _require_bundle()
    return {
        'features': b.features,
        'planet_types': b.planet_types,
        'planet_type_counts': b.planet_type_counts,
        'habitable_by_type': b.habitable_by_type,
        'medians': b.medians,
        'default_planet_type': b.default_planet_type,
        'metrics': b.metrics,
        'scatter': b.scatter,
        'grid': {'width': GRID_W, 'height': GRID_H},
    }


@app.post('/api/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    b = _require_bundle()
    if req.planet_type not in b.planet_types:
        raise HTTPException(422, f'Unknown planet type: {req.planet_type}')

    values = {
        'Planet_Mass': req.mass,
        'Planet_Radius': req.radius,
        'Planet_Surface_Temperature': req.surface_temperature,
        'Planet_Period': req.period,
        'Planet_Type': req.planet_type,
    }

    probability = b.predict(values)
    base_value, shap_values = b.explain(values)

    labels = {f['name']: f['label'] for f in b.features}
    labels['Planet_Type'] = 'Planet type'

    contributions = [
        ShapContribution(
            feature=name,
            label=labels.get(name, name),
            value=(float(values[name]) if name in b.numeric_cols else 0.0),
            contribution=sv,
            direction='up' if sv >= 0 else 'down',
        )
        for name, sv in zip(b.feature_cols, shap_values)
    ]
    contributions.sort(key=lambda c: abs(c.contribution), reverse=True)

    surface = None
    if req.include_surface and req.surface_x and req.surface_y:
        held = dict(values)
        if req.surface_hold is not None:
            if req.surface_hold.planet_type not in b.planet_types:
                raise HTTPException(422, f'Unknown planet type: {req.surface_hold.planet_type}')
            held['Planet_Mass'] = req.surface_hold.mass
            held['Planet_Radius'] = req.surface_hold.radius
            held['Planet_Type'] = req.surface_hold.planet_type
        surface = b.surface(held, req.surface_x, req.surface_y)

    return PredictResponse(probability=probability, base_value=base_value,
                           shap=contributions, surface=surface)


app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')


@app.get('/')
def index():
    return FileResponse(STATIC_DIR / 'index.html')
