from ThyroidInstance import ThyroidInstance
from numpy import array

train_attrib = []
train_targets = []

test_attrib = []
test_targets = []


def read_data(fname, set_type):
    with open(fname, "r") as f:
        for row in f:
            if set_type == "train":
                train_attrib.append(
                    [float(x) for x in row.strip().split()[:-1]])
                train_targets.append(
                    int(row.strip().split()[-1]))

            else:
                test_attrib.append(
                    [float(x) for x in row.strip().split()[:-1]])
                test_targets.append(
                    int(row.strip().split()[-1]))


def get_data(set_type):
    if set_type == "train":
        if not train_attrib:
            read_data("./data/ann-train.data", set_type)
        return (array(train_attrib), array(train_targets))

    if not test_attrib:
        read_data("./data/ann-test.data", set_type)
    return (array(test_attrib), array(test_targets))
