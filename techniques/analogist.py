from sklearn.svm import SVC


def result(train_data: tuple, test_data: tuple, verbose: bool):
    if verbose:
        print(">>>>>>>>>>>>>>>>>>>>>>>ANALOGIST<<<<<<<<<<<<<<<<<<<<<<<<<<")
    train_atts, train_targets = train_data
    test_atts, test_targets = test_data

    clf = SVC(verbose=verbose)

    y_predict = clf.fit(train_atts, train_targets).predict(test_atts)

    i = 0
    for predict, target in zip(y_predict, test_targets):
        if predict == target:
            i += 1

    return i
