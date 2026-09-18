"""Train and persist the 5-feature model that backs the interactive GUI.

The original pipeline in `prep_hist_model.py` trains on 24 features and is never
saved to disk -- `hist_gradient_boosting.classify()` refits it on every call.
The GUI needs (a) a persisted artifact it can load once at startup and (b) a
model whose inputs are exactly the five values the user manipulates, so the
sliders actually move the prediction.

Run:  python -m src.train_gui_model
"""

import json
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = pathlib.Path(__file__).resolve().parents[1]
HABITABLE_CSV = ROOT / 'data' / 'habitable_exoplanets.csv'
UNHABITABLE_CSV = ROOT / 'data' / 'unhabitable_exoplanets.csv'
ALL_CSV = ROOT / 'data' / 'all_exoplanets.csv'
ARTIFACT = ROOT / 'models' / 'habitability_gui_model.joblib'

NUMERIC_COLS = [
    'Planet_Mass',
    'Planet_Radius',
    'Planet_Surface_Temperature',
    'Planet_Period',
]
CATEGORICAL_COLS = ['Planet_Type']
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# Human-facing labels and units, keyed by the model's own column names.
FEATURE_META = {
    'Planet_Mass': ('Mass', 'M⊕', 'log'),
    'Planet_Radius': ('Radius', 'R⊕', 'linear'),
    # Log-scaled: the model's habitable band is ~233-320 K, a few percent of a
    # linear 233-2305 K axis, which would leave the slider dead across most of
    # its travel and smear the band into two grid rows.
    'Planet_Surface_Temperature': ('Surface temperature', 'K', 'log'),
    'Planet_Period': ('Orbital period', 'days', 'log'),
    'Planet_Type': ('Planet type', '', 'categorical'),
}


def load_training_frame():
    """Concatenate the two labelled CSVs exactly as the existing code does."""
    habitable = pd.read_csv(HABITABLE_CSV)
    unhabitable = pd.read_csv(UNHABITABLE_CSV)
    df = pd.concat([habitable, unhabitable], ignore_index=True)
    # The habitable file uses 1 and 2 for two flavours of habitability; the
    # original prepare_datasets.py collapses 2 -> 1, so we do the same.
    df['Planet_Habitability'] = df['Planet_Habitability'].replace(2, 1)
    return df


def build_pipeline():
    """Same estimator and preprocessing shape as src/prep_hist_model.py."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_COLS),
            ('cat', categorical_transformer, CATEGORICAL_COLS),
        ],
        remainder='passthrough',
    )
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(random_state=42)),
    ])


def slider_bounds(series):
    """Bound sliders by the 1st-99th percentile of real training values.

    The raw extremes are unusable for a slider: orbital period runs out to
    4.0e8 days and mass to 8.97e4 M(earth), so a full-range slider would spend
    almost all of its travel in a region containing a handful of planets.
    """
    clean = series.dropna()
    return {
        'min': float(clean.quantile(0.01)),
        'max': float(clean.quantile(0.99)),
        'true_min': float(clean.min()),
        'true_max': float(clean.max()),
        'median': float(clean.median()),
        'mean': float(clean.mean()),
        'p25': float(clean.quantile(0.25)),
        'p75': float(clean.quantile(0.75)),
        'null_fraction': float(series.isna().mean()),
    }


def scatter_points(df):
    """Real exoplanets for the background of the period/temperature plot."""
    cols = ['Planet_Name', 'Planet_Period', 'Planet_Surface_Temperature',
            'Planet_Type', 'Planet_Mass', 'Planet_Radius', 'Planet_Habitability']
    sub = df[cols].dropna(subset=['Planet_Period', 'Planet_Surface_Temperature'])
    points = []
    for row in sub.itertuples(index=False):
        points.append({
            'name': row.Planet_Name,
            'period': round(float(row.Planet_Period), 4),
            'temp': round(float(row.Planet_Surface_Temperature), 2),
            'type': None if pd.isna(row.Planet_Type) else str(row.Planet_Type),
            'mass': None if pd.isna(row.Planet_Mass) else round(float(row.Planet_Mass), 3),
            'radius': None if pd.isna(row.Planet_Radius) else round(float(row.Planet_Radius), 3),
            'habitable': int(row.Planet_Habitability > 0),
        })
    return points


def evaluate(x, y):
    """Honest cross-validated metrics -- the dataset is 1.25% positive, so a
    plain accuracy figure is meaningless here."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(build_pipeline(), x, y, cv=cv,
                              method='predict_proba')[:, 1]
    report = classification_report(y, (proba >= 0.5).astype(int),
                                   output_dict=True, zero_division=0)
    return {
        'cv_folds': 5,
        'roc_auc': float(roc_auc_score(y, proba)),
        'pr_auc': float(average_precision_score(y, proba)),
        'accuracy': float(report['accuracy']),
        'precision_habitable': float(report['1']['precision']),
        'recall_habitable': float(report['1']['recall']),
        'f1_habitable': float(report['1']['f1-score']),
        'n_samples': int(len(y)),
        'n_habitable': int(y.sum()),
        'positive_rate': float(y.mean()),
    }


def main():
    df = load_training_frame()
    x = df[FEATURE_COLS]
    y = df['Planet_Habitability'].astype(int)

    print(f'Training on {len(df)} planets '
          f'({int(y.sum())} habitable, {y.mean():.2%} positive rate)')

    metrics = evaluate(x, y)
    print(f"  CV ROC-AUC {metrics['roc_auc']:.3f}   PR-AUC {metrics['pr_auc']:.3f}")
    print(f"  habitable class: precision {metrics['precision_habitable']:.3f} "
          f"recall {metrics['recall_habitable']:.3f}")

    model = build_pipeline()
    model.fit(x, y)

    planet_types = sorted(df['Planet_Type'].dropna().unique().tolist())

    features = []
    for col in NUMERIC_COLS:
        label, unit, scale = FEATURE_META[col]
        features.append({
            'name': col, 'label': label, 'unit': unit,
            'scale': scale, **slider_bounds(df[col]),
        })

    all_planets = pd.read_csv(ALL_CSV)

    payload = {
        'model': model,
        'feature_cols': FEATURE_COLS,
        'numeric_cols': NUMERIC_COLS,
        'categorical_cols': CATEGORICAL_COLS,
        'features': features,
        'planet_types': planet_types,
        'planet_type_counts': {
            str(k): int(v) for k, v in df['Planet_Type'].value_counts().items()
        },
        'habitable_by_type': {
            str(k): int(v) for k, v in
            df[df['Planet_Habitability'] == 1]['Planet_Type']
            .value_counts().items()
        },
        'medians': {c: float(df[c].median()) for c in NUMERIC_COLS},
        'default_planet_type': df['Planet_Type'].mode()[0],
        'metrics': metrics,
        'scatter': scatter_points(all_planets),
        'sklearn_version': __import__('sklearn').__version__,
    }

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, ARTIFACT, compress=3)
    size_kb = ARTIFACT.stat().st_size / 1024
    print(f'Saved {ARTIFACT.relative_to(ROOT)} ({size_kb:.0f} KB, '
          f'{len(payload["scatter"])} scatter points)')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
