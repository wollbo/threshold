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


def exponential_mixture(span, p=0.5, l0=4, l1=6):
    f0 = np.exp(-l0 * span) * l0 / (1 - np.exp(-l0))
    f1 = - np.exp(l1 * span) * l1 / (1 - np.exp(l1))
    return p * f1 + (1-p) * f0, (1-p)*f0, p*f1


#def exponential_threshold(p=0.5, alpha=1, l1=5, l0=4):
#    return 1 + np.log((p / (p + alpha * (1-p))) ** (1/(l0+l1)))


def exponential_threshold(p=0.5, alpha=1, l=5, epsilon=0.001):
    q = 0.5
    q_new = 1
    while abs(q-q_new)>epsilon:
        q = q_new
        f, d = exponential_integral(p, q, l)
        q_new = q - (f-kl_threshold(p, alpha))/d
    return q


def exponential_integral(p=0.5, q=0.5, l=5):
    f_val = np.exp(-l*q) * (np.exp(l)-np.exp(l*q)) * (p*(np.exp(l*q)-1) + 1) / (np.exp(l)-1)
    df_val = np.exp(-l*q) * (l*np.exp(l)*(p-1) - l*p*np.exp(2*l*q)) / (np.exp(l)-1)
    return f_val, df_val