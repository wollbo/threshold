import numpy as np
import tensorflow as tf
import pandas as pd
import sys
from sklearn import datasets, metrics


def load_data(dataset="breastcancer", return_X_y=True):

    if dataset == "breastcancer":
        (data, target) = datasets.load_breast_cancer(return_X_y=return_X_y)
    elif dataset == "digits":
        (data, target) = datasets.load_digits(return_X_y=return_X_y)
    elif dataset == "credit":
        cdt = datasets.fetch_openml(name='credit-g')
        data = cdt["data"]
        target = np.array([1 if item == 'good' else 0 for item in cdt["target"]])
    else:
        print("Invalid dataset")
        sys.exit(1)
    return data, target


def kl_threshold(p, alpha):
    return p / (p + alpha * (1-p))


def kl_cost(span, p=0.5, alpha=1):
    return - p * np.log(span) - alpha * (1-p) * np.log(1-span)


def normalize_hist(values, hist):
    for item in hist:
        item.set_height(item.get_height()/sum(values))
