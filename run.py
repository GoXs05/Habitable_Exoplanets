import argparse

from src import hist_gradient_boosting_tester as hgbt

def parse_args():

    supported_categorical_vars = ["['Planet_Type', 'Planet_Temperature_Type', 'Star_Temperature_Type']"]
    parser = argparse.ArgumentParser(description='Exoplanet Habitability')
    parser.add_argument('--categorical_vars', default="MiDaS_small", choices=supported_categorical_vars, help='select the categorical variables')
    args = parser.parse_args()
    return args

if  __name__ == '__main__':
    args = parse_args()
    hgbt.test_hist_gradient_boosting()