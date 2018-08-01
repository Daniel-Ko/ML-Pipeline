import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def full_pipeline(training_data, test_data, threshold, verbose):
    piped_training_data = training_data
    piped_test_data = test_data

    # OUTLIER PRUNING. RUNS ON BOTH TRAINING AND TEST DATA
    if verbose:
        print("========================\nOutlier Pruning on entire dataset\n========================")
    piped_training_data, piped_test_data = prune_outliers_from_dataset(
        (piped_training_data, piped_test_data), threshold, verbose)

    # DIMENSION REDUCTION
    if verbose:
        print("\n========================\nTRAINING Dimension Reduction\n========================")
    piped_training_data = reduce_dims(piped_training_data, threshold, verbose)

    if verbose:
        print(
            "\n========================\nTEST Dimension Reduction\n========================")
    piped_test_data = reduce_dims(
        piped_test_data, threshold, verbose, principal_components=len(piped_training_data[0][0]))

    piped_training_data = (
        np.array(piped_training_data[0]), np.array(piped_training_data[1]))
    piped_test_data = (
        np.array(piped_test_data[0]), np.array(piped_test_data[1]))

    return (piped_training_data, piped_test_data,)


def prune_outliers_from_dataset(datasets: tuple, threshold: float, verbose: bool) -> tuple:
    """
    Detects outliers from training and test instances and prunes them from the corresponding attribute/target lists.

    Params:
        datasets (tuple): training and test datasets (each dataset is a tuple of attribute + target lists)
        threshold (float): "confidence level" to determine outlier region
        verbose (bool): whether to print control flow and pruning messages

    Note that both datasets are passed in at once. This is because test dataset must be pruned to the exact same shape as training's.
    If training after pruning turns out to be longer than test after prune, training will be reduced to test's size instead.

    Returns:
        (tuple) a pruned datasets. See datasets param. Same format, just pruned.
    """
    # Printing pruning control flow
    outlier_frac = 1.0 - threshold

    non_outlier_datasets = []
    pruned_training_size = 0

    for dataset in datasets:
        outliers_predict = EllipticEnvelope(
            contamination=outlier_frac).fit(dataset[0]).predict(dataset[0])

        pruned_atts, pruned_targs = [], []  # = data

        for i, pred in enumerate(outliers_predict):
            if pred != -1:
                pruned_atts.append(dataset[0][i])
                pruned_targs.append(dataset[1][i])

        non_outlier_data = (pruned_atts, pruned_targs)

        # Make certain that both training and test will have the same shape when modeling!

        size_outliers = len(pruned_targs)

        # Training set's outliers
        if pruned_training_size == 0:
            pruned_training_size = len(non_outlier_data[0])

            if verbose:
                print(
                    f"initalised training non-outliers to size {size_outliers}")

        # If test set's # of non-outliers are smaller than training set's, prune training set empathetically
        elif pruned_training_size > size_outliers:
            test_shape = size_outliers
            training_data_to_prune_more = non_outlier_datasets[0]

            non_outlier_datasets[0] = (
                training_data_to_prune_more[0][:test_shape], training_data_to_prune_more[1][:test_shape])

            pruned_training_size = len(non_outlier_data[0][0])

            if verbose:
                print(f"pruning training non-outliers to size {test_shape}")

        # Any other case (test outliers greater than training, or UNLIKELY, if equal)
        else:
            non_outlier_data = (
                non_outlier_data[0][:pruned_training_size], non_outlier_data[1][:pruned_training_size])

            if verbose:
                print(
                    f"pruning test non-outliers to size {pruned_training_size}")

        # Add finished dataset to the complete dataset: (training data, test data)
        non_outlier_datasets.append(non_outlier_data)

    return non_outlier_datasets


def reduce_dims(data: tuple, threshold: float, verbose: bool, principal_components: int =0) -> tuple:
    """
    Dimension reduction on one dataset. The data is standardised first, then reduced. 
    The training data will be reduced to a certain confidence threshold, and the test will be reduced to the remembered
    principal components size.

    Params:
        data (tuple): Holds attribute and target lists
        threshold (float): "confidence level", AKA once the running total of 
                    components collect this threshold of explained variance, no more components are added.
        verbose (bool): whether to print component collection progress
        principal_components(int): a remembered components to reduce to (if it isn't 0)

    Returns:
        data (see params) with reduced components
    """
    # Standardising
    normal_atts = StandardScaler().fit_transform(
        data[0], data[1])

    # Determining how many components to reduce to (confidence lvl set in options)
    pca = PCA(svd_solver='auto')
    reduced_atts = pca.fit_transform(normal_atts)

    if principal_components == 0:
        # number of components that would push to set confidence level
        set_principal_comps = len(normal_atts[0])
        total_explained_var = 0.0

        for i, ratio in enumerate(pca.explained_variance_ratio_):
            total_explained_var += ratio

            # if TEVar is above threshold,  reduce to this many components
            if total_explained_var > threshold and i < len(normal_atts[0])-1:
                set_principal_comps = i+1
                break

        principal_components = set_principal_comps
    else:
        set_principal_comps = principal_components

    # Reduce
    pca = PCA(svd_solver='auto', n_components=set_principal_comps)
    reduced_atts = pca.fit_transform(normal_atts)

    if verbose:
        print_PCA_variance_ratios(pca.explained_variance_ratio_)
    return (reduced_atts, data[1])


def print_PCA_variance_ratios(ratios):
    print("PCA DIMENSIONALITY REDUCTION")
    print(f">> Reduced to {len(ratios)} dimensions")

    total_explained_var = 0.0
    for i, ratio in enumerate(ratios):
        total_explained_var += ratio

        print(f"\tFeature {i}: (ratio={0:.5f}", end=", ")
        print("Explained variance so far= {0:.3f}%)".format(
            total_explained_var*100))
