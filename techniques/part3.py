import sys
from os.path import dirname, abspath
import argparse

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# DataReader is outside this "partX" directory
sys.path.append(dirname(dirname(abspath(__file__))))
import DataReader
import Pipeline
import analogist
import bayesian
import connectionist
import evolutionist
import symbolist


def main():

    # Add arguments and parameters
    parser = argparse.ArgumentParser(
        description='Determine which program(s) to run and how verbose reports should be')

    parser.add_argument("-a", "--ana", action='store_true',
                        help="analogist results")
    parser.add_argument("-b", "--bay", action='store_true',
                        help="bayesian results")
    parser.add_argument("-c", "--conn", action='store_true',
                        help="connectionist results")
    parser.add_argument("-e", "--evo", action='store_true',
                        help="evolutionist results")
    parser.add_argument("-s", "--symb", action='store_true',
                        help="symbolist results")
    parser.add_argument("-A", "--all", action='store_true', help="all results")
    parser.add_argument(
        "-v", "--verbose", action='store_true', help="print verbose options, if any (Recommend with one tribe at a time)")
    parser.add_argument("-pv", "--pipeline-verbose",
                        action='store_true', help="Print pipeline reasoning")

    parser.add_argument("threshold", nargs='?', type=float, const=0.85, default=0.85,
                        help="Baseline accuracy wanted for pipeline transformers")
    parser.add_argument("runs", nargs='?', type=int, const=5,
                        default=5, help="Number of runs to average accuracy over")

    args = parser.parse_args()

    # receive data
    train_data = DataReader.get_data("train")
    test_data = DataReader.get_data("test")

    # run data through the pipeline
    full_dataset = Pipeline.full_pipeline(
        train_data, test_data, threshold=args.threshold, verbose=args.pipeline_verbose)

    test_size = full_dataset[0][0].shape[0]

    # run ML techniques
    if args.verbose or args.pipeline_verbose:
        print()
    print("========================\nRESULTS\n========================")
    if(args.ana or args.all):
        average_accuracy(
            model=analogist.result, runs=args.runs, modelName="SUPPVEC",
            datasets=full_dataset, verbose=args.verbose, test_size=test_size)
    if(args.bay or args.all):
        average_accuracy(
            model=bayesian.result, runs=args.runs, modelName="GAUSSNAIVEBAYES",
            datasets=full_dataset, verbose=args.verbose, test_size=test_size)
    if(args.conn or args.all):
        average_accuracy(
            model=connectionist.result, runs=args.runs, modelName="MLP",
            datasets=full_dataset, verbose=args.verbose, test_size=test_size)
    if(args.evo or args.all):
        average_accuracy(
            model=evolutionist.result, runs=args.runs, modelName="EVOLSYMBREGR",
            datasets=full_dataset, verbose=args.verbose, test_size=test_size)
    if(args.symb or args.all):
        average_accuracy(
            model=symbolist.result, runs=args.runs, modelName="DECTREE",
            datasets=full_dataset, verbose=args.verbose, test_size=test_size)


def average_accuracy(datasets: tuple, model, runs: int, modelName: str, verbose: bool, test_size: int):
    total_correct = 0
    for i in range(runs):
        verbose = True if i == runs-1 else False
        total_correct += model(*datasets, verbose)

    total_correct /= runs
    print("{}: {}/{} correct, {:.2f}%".format(modelName, total_correct,
                                              test_size, 100*total_correct/test_size))


if __name__ == "__main__":
    main()
