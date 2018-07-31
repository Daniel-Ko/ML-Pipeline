from gplearn.genetic import SymbolicRegressor


def result(train_data: tuple, test_data: tuple, verbose: bool):
    train_atts, train_targets = train_data
    test_atts, test_targets = test_data

    clf = SymbolicRegressor(verbose=verbose)

    y_predict = clf.fit(train_atts, train_targets).predict(test_atts)

    i = 0
    for predict, target in zip(y_predict, test_targets):
        if predict == target:
            i += 1

    if verbose:
        print(F"FINAL PROGRAM: {clf._program}")

    print("EVO: {}/{} correct, {:.2f}%".format(i,
                                               test_atts.shape[0], 100*i/test_atts.shape[0]))
