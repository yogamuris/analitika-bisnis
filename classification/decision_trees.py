import csv
from collections import defaultdict

import numpy as np
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log

import pprint

def csv_to_dict(filename, columns):
    """
    fungsi untuk mengubah csv ke dalam struktur data dict.

    Args:
        filename: Nama file csv.
        columns: Kolom dari data.

    Returns:
        dataset dalam struktur data dict.
    """

    data = defaultdict(list)
    with open(filename, newline='') as csvfile:
        header = csv.DictReader(csvfile, columns)
        next(header)
        for row in header:
            for field in columns:
                data[field].append(row[field])
    return dict(data)

def calculate_entropy(data):
    """
    fungsi untuk menghitung entropy dari atrribut (Entropy(S)).

    Args:
        data: training data.

    Returns:
        nilai Entropy(S).
    """

    Class = data.keys()[-1]
    entropy = 0
    class_values = data[Class].unique()
    for value in class_values:
        p = data[Class].value_counts()[value]/len(data[Class])
        entropy += -p*np.log2(p)
    
    return entropy

def calculate_entropy_attribute(data, attribute):
    """
    fungsi untuk menghitung entropy dari branch attribut (Entropy(Sv)).

    Args:
        data: training data.
        attribute: attribute dari data.

    Returns:
        nilai Entropy(Sv).
    """

    Class = data.keys()[-1]
    class_values = data[Class].unique()
    branch_values = data[attribute].unique()
    entropy2 = 0
    for branch_value in branch_values:
        entropy = 0
        for class_value in class_values:
            num = len(data[attribute][data[attribute] == branch_value][data[Class]==class_value])
            den = len(data[attribute][data[attribute] == branch_value])
            fraction = num/(den+eps)
            entropy += -fraction*log(fraction+eps)
        fraction2 = den/len(data)
        entropy2 += -fraction2*entropy

    return abs(entropy2) 

def find_higher_gain(data):
    """
    fungsi untuk mencari attribute yang memiliki Gain terbesar.

    Args:
        data: training data.

    Returns:
        attribut yang memiliki Gain(S, attribute) terbesar.
    """

    entropy_attribute = data.keys()[:-1]
    gain = []
    for attribute in entropy_attribute:
        gain.append(calculate_entropy(data)-calculate_entropy_attribute(data, attribute))
    
    return data.keys()[:-1][np.argmax(gain)]

def get_branch_value(data, node, value):
    """
    fungsi untuk mencari nilai dari branch dari attribut.

    Args:
        data: training data.
        node: attribut yang menjadi decision node.
        value: nilai dari branch node.

    Returns:
        daftar nilai dari branch.
    """

    return data[data[node] == value].reset_index(drop=True)

def create_tree(data, tree=None):
    """
    fungsi untuk membuat permodelan tree.

    Args:
        data: training data.
        tree: tree.

    Returns:
        permodelan tree dengan struktur data dict.
    """

    node = find_higher_gain(data)

    node_branch_value = np.unique(data[node])

    if tree is None:
        tree = {}
        tree[node] = {}
    
    for value in node_branch_value:
        branch_value = get_branch_value(data, node, value)
        class_value, counts = np.unique(branch_value['label'], return_counts=True)

        if len(counts) == 1:
            tree[node][value] = class_value[0]
        else:
            tree[node][value] = create_tree(branch_value)

    return tree

def predict(data_test, tree):
    """
    fungsi untuk membuat prediksi output label dari test data.

    Args:
        data_test: data untuk di test.
        tree: tree.

    Returns:
        nilai dari Class.
    """
    
    for nodes in tree.keys():
        value = data_test[nodes]
        tree = tree[nodes][value]
        prediction = 0

        if type(tree) is dict:
            prediction = predict(data_test, tree)
        else:
            prediction = tree
            break

    return prediction


if __name__ == "__main__":
    # Load dataset
    dataset = csv_to_dict('disease.csv', columns=['Age', 'Gender', 'BMI', 'Intensity', 'label'])
    data = pd.DataFrame(dataset, columns=['Age', 'Gender', 'BMI', 'Intensity', 'label'])
    
    # create tree
    tree = create_tree(data)
    
    # data test to predict
    data_test = {'Age':'young', 'Gender':'male'}
    
    # prediction
    prediction = predict(data_test, tree)

    # output
    print("Decision Tree:")
    pprint.pprint(tree)
    print("\nPrediksi : ",prediction)