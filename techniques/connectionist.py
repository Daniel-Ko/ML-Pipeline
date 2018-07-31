from sklearn.neural_network import MLPClassifier


def result(train_data: tuple, test_data: tuple, verbose: bool):
    train_atts, train_targets = train_data
    test_atts, test_targets = test_data

    clf = MLPClassifier(max_iter=500, verbose=verbose)

    y_predict = clf.fit(train_atts, train_targets).predict(test_atts)

    i = 0
    for predict, target in zip(y_predict, test_targets):
        if predict == target:
            i += 1

    print("MLP: {}/{} correct, {:.2f}%".format(i,
                                               test_atts.shape[0], 100*i/test_atts.shape[0]))
    if verbose:
        print("\tTook {} iterations. {} outputs. {} layers".format(
            clf.n_iter_, clf.n_outputs_, clf.n_layers_))
