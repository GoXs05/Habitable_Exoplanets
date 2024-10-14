from src import prepare_datasets as data
from src import prep_hist_model as model

def classify(user_input_df):
    habitable = 'data/habitable_exoplanets.csv'
    unhabitable = 'data/unhabitable_exoplanets.csv'
    x_train, y_train, x_test, categorical_cols, numeric_cols = data.prep_user_data(habitable, unhabitable, user_input_df)

    clf = model.prep_model(categorical_cols, numeric_cols)

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    if y_pred == [1]:
        print("Habitable")
    else:
        print("Unhabitable")