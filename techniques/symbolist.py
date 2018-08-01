from sklearn.tree import DecisionTreeClassifier, export_graphviz


def result(train_data: tuple, test_data: tuple, verbose: bool):
    if verbose:
        print(">>>>>>>>>>>>>>>>>>>>>>>SYMBOLIST<<<<<<<<<<<<<<<<<<<<<<<<<<")
    train_atts, train_targets = train_data
    test_atts, test_targets = test_data

    clf = DecisionTreeClassifier()

    y_predict = clf.fit(train_atts, train_targets).predict(test_atts)

    i = 0
    for predict, target in zip(y_predict, test_targets):
        if predict == target:
            i += 1

    if verbose:
        print(
            f"FeatImportance: {clf.feature_importances_}, Features: {clf.n_features_}, Outputs:{clf.n_outputs_}")

        # To export tree data
        # export_graphviz(clf, class_names=['1', '2', '3'])
    return i
