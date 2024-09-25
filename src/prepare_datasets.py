import pandas as pd

def prep_data(habitable_data, unhabitable_data):
    habitable = pd.read_csv(habitable_data)
    unhabitable = pd.read_csv(unhabitable_data)

    df = pd.concat([habitable, unhabitable], ignore_index=True)
    df ['Planet_Habitability'] = df['Planet_Habitability'].replace(2, 1)

    # Drop columns that are irrelevant to the model
    x = df.drop(['Planet_Name', 'Planet_Detection_Method', 'Planet_Habitability_Zone_Location', 'Star_Name', 'Planet_Habitability'], axis=1)
    y = df['Planet_Habitability']

    # Handle categorical variables by converting into dummy variables
    categorical_cols = ['Planet_Type', 'Planet_Temperature_Type', 'Star_Temperature_Type']

    return x, y, categorical_cols