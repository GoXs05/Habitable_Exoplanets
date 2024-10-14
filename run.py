import argparse

from src import hist_gradient_boosting_tester as hgbt
from src import user_input
from src import hist_gradient_boosting as hgb


def parse_args():
    parser = argparse.ArgumentParser(description='Exoplanet Habitability')
    parser.add_argument('-test', default=False, action='store_const', const=True, help='Test the model?')
    parser.add_argument('-report', default=False, action='store_const', const=True, help='If testing, print classification report?')
    args = parser.parse_args()
    return args

if  __name__ == '__main__':
    args = parse_args()
    if args.test:
        hgbt.train_hist_gradient_boosting(args.report, test_size_input=0.2)
    else: 
        df = user_input.run_UI()
        hgb.classify(df)