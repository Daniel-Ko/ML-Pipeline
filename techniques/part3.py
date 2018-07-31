import sys
from os.path import dirname, abspath
import argparse

from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# DataReader is outside this "partX" directory
sys.path.append(dirname(dirname(abspath(__file__))))
import DataReader
import analogist
import bayesian
import connectionist
import evolutionist
import symbolist

# Pipe verbosity output?
pipe_verbose = False
# The baseline accuracy for pipeline functions
pipe_accuracy = 0.85
# Remember shape of training data after outliers have been pruned, and apply to test. Store list
# in case we need to prune training to test's size
global_pruned_outliers = []
# Makes sure training/test use same shape of reduction (training sets this first)
principal_components = 0


def main():
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

    parser.add_argument("pipe_accuracy", nargs='?', type=float, const=0.85, default=0.85,
                        help="Baseline accuracy wanted for pipeline transformers")

    args = parser.parse_args()

    # receive data
    train_data = DataReader.get_data("train")
    test_data = DataReader.get_data("test")

    # set option for verbose pipeline reports
    global pipe_verbose
    pipe_verbose = args.pipeline_verbose

    # set baseline of pipeline accuracy
    global pipe_accuracy
    pipe_accuracy = args.pipe_accuracy

    # run data through the pipeline
    print("\n========================\nTRAINING PIPELINE\n========================")
    piped_train_data = pipe1(train_data)
    print("\n========================\nTEST PIPELINE\n========================")
    piped_test_data = pipe1(test_data)
    print("\n========================\nTRAINING PIPELINE\n========================")
    piped_train_data = pipe2(piped_train_data)
    print("\n========================\nTEST PIPELINE\n========================")
    piped_test_data = pipe2(piped_test_data)

    # import matplotlib.pyplot as plt
    # import matplotlib
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.set_zlabel("x_composite_3")

    # # Plot the compressed data points
    # ax.scatter(piped_train_data[0][:, 0], piped_train_data[0][:, 1],
    #            zs=piped_train_data[0][:, 2], s=4, lw=0, label="inliers")

    # # Plot x's for the ground truth outliers
    # # ax.scatter(piped_train_data[0][:, 0], piped_train_data[0][:, 1], zs=piped_train_data[0][:, 2],
    # #            lw=2, s=60, marker="x", c="red", label="outliers")
    # ax.legend()
    # plt.show()

    # run ML techniques
    print("\n========================\nRESULTS\n========================")
    if(args.ana or args.all):
        analogist.result(piped_train_data, piped_test_data, args.verbose)
    if(args.bay or args.all):
        bayesian.result(piped_train_data, piped_test_data, args.verbose)
    if(args.conn or args.all):
        connectionist.result(piped_train_data, piped_test_data, args.verbose)
    if(args.evo or args.all):
        evolutionist.result(piped_train_data, piped_test_data, args.verbose)
    if(args.symb or args.all):
        symbolist.result(piped_train_data, piped_test_data, args.verbose)


def pipe1(data):
    piped_data = data
    piped_data = prune_outliers(piped_data)
    return piped_data


def pipe2(data):
    piped_data = data
    piped_data = reduce_dims(piped_data)
    return piped_data


def prune_outliers(data):
    # Printing pruning control flow
    global pipe_verbose

    global pipe_accuracy
    outlier_frac = 1.0 - pipe_accuracy

    outliers_predict = EllipticEnvelope(
        contamination=outlier_frac).fit(data[0]).predict(data[0])

    pruned_atts, pruned_targs = [], []  # = data

    for i, pred in enumerate(outliers_predict):
        if pred != -1:
            pruned_atts.append(data[0][i])
            pruned_targs.append(data[1][i])

    non_outlier_data = (pruned_atts, pruned_targs)
    # pruned_atts, pruned_targs, non_outliers=zip(
    #     *((x, y, z) for x, y, z in zip(pruned_atts, pruned_targs, outliers_predict) if lambda z: z != -1))
    # print(non_outliers)
    # non_outliers = [x for x in outliers_predict if x != -1]
    # print(len(non_outliers))

    # Make certain that both training and test will have the same shape when modeling!
    global global_pruned_outliers

    size_outliers = len(pruned_targs)

    # Training set's outliers
    if not global_pruned_outliers:
        global_pruned_outliers = non_outlier_data

        if pipe_verbose:
            print(
                f"initalised training non-outliers to size {size_outliers}")

    # If test set's # of non-outliers are smaller than training set's, prune training set empathetically
    elif len(global_pruned_outliers[1]) > size_outliers:
        test_shape = size_outliers
        global_pruned_outliers = (
            global_pruned_outliers[0][:test_shape], global_pruned_outliers[1][:test_shape])

        if pipe_verbose:
            print(f"pruning training non-outliers to size {test_shape}")

    # Any other case (test outliers greater than training, or UNLIKELY, if equal)
    else:
        training_shape = len(global_pruned_outliers[1])
        non_outlier_data = (
            non_outlier_data[0][:training_shape], non_outlier_data[1][:training_shape])

        if pipe_verbose:
            print(f"pruning test non-outliers to size {training_shape}")

    return non_outlier_data


def reduce_dims(data):

    # Standardising
    normal_atts = StandardScaler().fit_transform(
        data[0], data[1])

    # Determining how many components to reduce to (confidence lvl set in options)
    pca = PCA(svd_solver='auto')
    reduced_atts = pca.fit_transform(normal_atts)

    global principal_components

    if principal_components == 0:
        # number of components that would push to set confidence level
        set_principal_comps = len(normal_atts[0])
        total_explained_var = 0.0

        for i, ratio in enumerate(pca.explained_variance_ratio_):
            total_explained_var += ratio
            global pipe_accuracy
            # if TEVar is above threshold,  reduce to this many components
            if total_explained_var > pipe_accuracy and i < len(normal_atts[0])-1:
                set_principal_comps = i+1
                break

        principal_components = set_principal_comps
    else:
        set_principal_comps = principal_components

    # Reduce
    pca = PCA(svd_solver='auto', n_components=set_principal_comps)
    reduced_atts = pca.fit_transform(normal_atts)

    global pipe_verbose

    if pipe_verbose:
        print_PCA_variance_ratios(pca.explained_variance_ratio_)
    return (reduced_atts, data[1])


def print_PCA_variance_ratios(ratios):
    print("PCA DIMENSIONALITY REDUCTION")
    total_explained_var = 0.0

    for i, ratio in enumerate(ratios):
        total_explained_var += ratio

        print(f"\tFeature {i}: (ratio={0:.5f}", end=", ")
        print("Explained variance so far= {0:.3f}%)".format(
            total_explained_var*100))


if __name__ == "__main__":
    main()
