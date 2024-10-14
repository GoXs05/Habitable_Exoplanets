import pandas as pd

def prep_data(habitable_data, unhabitable_data):
    habitable = pd.read_csv(habitable_data)
    unhabitable = pd.read_csv(unhabitable_data)

    df = pd.concat([habitable, unhabitable], ignore_index=True)
    df['Planet_Habitability'] = df['Planet_Habitability'].replace(2, 1)

    # Drop columns that are irrelevant to the model
    x = df.drop(['Planet_Name', 'Planet_Detection_Method', 'Planet_Habitability_Zone_Location', 'Star_Name', 'Planet_Habitability'], axis=1)
    y = df['Planet_Habitability']

    # Handle categorical variables by converting into dummy variables
    categorical_cols = ['Planet_Type', 'Planet_Temperature_Type', 'Star_Temperature_Type']
    numeric_cols = x.select_dtypes(include=['float64', 'int64']).columns.tolist()

    return x, y, categorical_cols, numeric_cols

def prep_user_data(habitable_data, unhabitable_data, user_df):
    habitable = pd.read_csv(habitable_data)
    unhabitable = pd.read_csv(unhabitable_data)

    df = pd.concat([habitable, unhabitable], ignore_index=True)
    df['Planet_Habitability'] = df['Planet_Habitability'].replace(2, 1)

    # Drop columns that are irrelevant to the model
    x_train = df.drop(['Planet_Name', 'Planet_Detection_Method', 'Planet_Habitability_Zone_Location', 'Star_Name', 'Planet_Habitability'], axis=1)
    y_train = df['Planet_Habitability']
    
    x_test = user_df.drop(['Planet_Habitability_Zone_Location', 'Star_Name', 'Planet_Habitability'], axis=1)

    # Handle categorical variables by converting into dummy variables
    categorical_cols = ['Planet_Type', 'Planet_Temperature_Type', 'Star_Temperature_Type']
    numeric_cols = x_train.select_dtypes(include=['float64', 'int64']).columns.tolist()

    return x_train, y_train, x_test, categorical_cols, numeric_cols