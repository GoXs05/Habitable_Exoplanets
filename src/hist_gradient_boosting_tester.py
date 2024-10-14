from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src import prepare_datasets as data
from src import prep_hist_model as model


def train_hist_gradient_boosting(print_report, test_size_input):
    habitable = 'data/habitable_exoplanets.csv'
    unhabitable = 'data/unhabitable_exoplanets.csv'
    x, y, categorical_cols, numeric_cols = data.prep_data(habitable, unhabitable)

    clf = model.prep_model(categorical_cols, numeric_cols)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_input, random_state=42)

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print(y_pred)

    if print_report:
        print(classification_report(y_test, y_pred))