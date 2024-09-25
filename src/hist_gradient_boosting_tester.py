import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from src import prepare_datasets as prep

def test_hist_gradient_boosting():
    # Handle categorical variables by converting into dummy variables
    habitable = 'data/habitable_exoplanets.csv'
    nonhabitable = 'data/unhabitable_exoplanets.csv'
    x, y, categorical_cols = prep.prep_data(habitable, nonhabitable)

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ], remainder='passthrough')

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier())
    ])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))