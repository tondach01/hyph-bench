import argparse

from hyperparameters import stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="", help="Path to file to analyse")
    parser.add_argument("-d", action="store_true", help="Trigger dataset analysis")
    parser.add_argument("-t", action="store_true", help="Output in tabular format")
    parser.add_argument("--header", action="store_true", help="Print out just header and return (relevant for tabular output)")
    args = parser.parse_args()

    if args.d:
        print(stats.DatasetInfo(args.file).report(tabular=args.t, header=args.header))