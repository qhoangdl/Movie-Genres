"""
This module include several functions used for evaluation
    - jaccard(): calculates the jaccard index
        args: two sets a and b
        returns: the jaccard index
    - confusion_matrix_stats(): calculate and print accuracy, precision, recall and F-score
        args:
            TP, TN, FP, FN: float values for True Positive, True Negative, False Positive and False Negative
            Genre: a string of genre name
            output: if True, then print the results
            file: if is not None, then print to the file also
        returns:
            accuracy, precision, recall and F-score
    - eval(): evaluate jaccard, accuracy, precision, recall and F-score for each genre
        args:
            test_genres: a list of list of correct genres for each test example
            predict: a list of list of predicted genres for each test example
            labels: the list of genre names
            file: if is not None, then print the results to file
        returns:
            jaccards, TP, FP, TN, FN: dictionaries of corresponding statistics for each genre
    - predict_dist(predict, file=None): print size distribution of the predicted genre sets
        args:
            predict: the list of list of predicted genres for each test exmaple
            file: if is not None, then print the results to file
"""

import numpy as np
from collections import defaultdict

def jaccard(a, b):
    return len(a.intersection(b)) * 1.0 / len(a.union(b))


def confusion_matrix_stats(TP, TN, FP, FN, genre, output=True, file=None):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if abs(TP + FP) > 1e-3 else 0.0
    recall = TP / (TP + FN)
    F1 = 2.0 * precision * recall / (precision + recall) if abs(precision + recall) > 1e-3 else 0.0

    if output:
        print("--------- {} Genre ---------".format(genre))
        print("Accuracy: {:.2f}%".format(accuracy * 100.0))
        print("Precision: {:.2f}%".format(precision * 100.0))
        print("Recall: {:.2f}%".format(recall * 100.0))
        print("F1: {:.2f}".format(F1))

        print("--------- {} Genre ---------".format(genre), file=file)
        print("Accuracy: {:.2f}%".format(accuracy * 100.0), file=file)
        print("Precision: {:.2f}%".format(precision * 100.0), file=file)
        print("Recall: {:.2f}%".format(recall * 100.0), file=file)
        print("F1: {:.2f}".format(F1), file=file)

    return accuracy, precision, recall, F1


def eval(test_genres, predict, labels, file=None):
    TP = defaultdict(float)
    FP = defaultdict(float)
    TN = defaultdict(float)
    FN = defaultdict(float)
    jaccards = []
    for x, y in zip(test_genres, predict):
        target_genres = set(x)
        predict_genres = set(y)
        jaccards.append(jaccard(target_genres, predict_genres))

        for genre in labels:
            if genre in predict_genres:
                if genre in target_genres:
                    TP[genre] += 1
                else:
                    FP[genre] += 1
            else:
                if genre in target_genres:
                    FN[genre] += 1
                else:
                    TN[genre] += 1

    jaccards = np.array(jaccards)

    print("Average Jaccard index: {:.2f}%".format(np.mean(jaccards) * 100.0))
    print("Average Jaccard index: {:.2f}%".format(np.mean(jaccards) * 100.0), file=file)
    for genre in labels:
        confusion_matrix_stats(TP[genre], TN[genre], FP[genre], FN[genre], genre, file=file)

    confusion_matrix_stats(sum(TP.values()), sum(TN.values()), sum(FP.values()), sum(FN.values()), "all", file=file)

    return jaccards, TP, FP, TN, FN


def predict_dist(predict, file=None):
    N = len(predict)
    count = defaultdict(float)
    for x in predict:
        count[len(x)] += 1.0

    for x in sorted(count.keys()):
        print("{:02d}: {:.02f}%".format(x, count[x] * 100.0 / N))
        if file is not None:
            print("{:02d}: {:.02f}%".format(x, count[x] * 100.0 / N), file=file)