from gplearn.genetic import SymbolicRegressor


def result(train_data: tuple, test_data: tuple, verbose: bool):
    if verbose:
        print(">>>>>>>>>>>>>>>>>>>>>>>EVOLUTIONIST<<<<<<<<<<<<<<<<<<<<<<<<<<")

    train_atts, train_targets = train_data
    test_atts, test_targets = test_data

    clf = SymbolicRegressor(verbose=verbose)

    y_predict = clf.fit(train_atts, train_targets).predict(test_atts)

    i = 0
    for predict, target in zip(y_predict, test_targets):
        if predict == target:
            i += 1

    if verbose:
        print(F"\tFINAL PROGRAM: {clf._program}")

    return i
