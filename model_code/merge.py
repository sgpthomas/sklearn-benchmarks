import pandas as pd
import argparse
import glob

def make_parser():
    descr = "Merges all .pkl files into a single .pkl file."
    parser = argparse.ArgumentParser(description=descr)

    parser.add_argument('resultsdir', action="store")
    parser.add_argument('output_name', action="store")

    return parser

def merge_all(resultsdir):
    files = glob.glob("{}/*.pkl".format(resultsdir))
    df = pd.DataFrame()
    for f in files:
        df = pd.concat([df, pd.read_pickle(f)])
    return df.sort_values('dataset').reset_index(drop=True)

if __name__ == "__main__":
    parser = make_parser().parse_args()

    combined = merge_all(parser.resultsdir)
    combined.to_pickle(parser.output_name)