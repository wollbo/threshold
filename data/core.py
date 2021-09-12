import numpy as np
import pandas as pd
import sys
import math
from sklearn import datasets
from scipy.stats import norm


def load_data(dataset="breastcancer", return_X_y=True):
    if dataset == "breastcancer":
        (data, target) = datasets.load_breast_cancer(return_X_y=return_X_y)
    elif dataset == "credit":
        cdt = datasets.fetch_openml(name='credit-g')
        data = cdt["data"]
        target = np.array([1 if item == 'good' else 0 for item in cdt["target"]])
    elif dataset == "orange-small":  # https://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data
        # using training dataset as complete dataset due to true test labels not being available
        raw = pd.read_csv('data/dataset/orange-small/orange_small_train.data', sep="\t")
        data = clean_data(raw)
        target = pd.read_csv('data/dataset/orange-small/labels/orange_small_train_churn.labels', sep="\t", header=None)
        target.to_numpy()
        target[target < 0] = 0
        target = np.squeeze(target)
    elif dataset == "orange-large":  # will probably not be possible to train models in a reasonable amount of time
        print("Not implemented yet!")
        sys.exit(1)  # add functionality for reading data batches and cleaning with SQL
    else:
        print("Invalid dataset")
        sys.exit(1)
    return data, target


def clean_data(data):
    """ Takes a DataFrame with incomplete values and fills in NaN fields, dropping constant columns, returning array """

    data.fillna(value=data.mean(), inplace=True)  # fills numerical NaN values with mean
    data.fillna(value=str(0), inplace=True)  # fills categorical NaN values with zero string
    data.loc[:, (data != data.iloc[0]).any()]  # drops constant columns, may cause errors in train/test indexing
    categorical = data.select_dtypes(['object']).columns
    for cat in categorical:
        data[cat] = data[cat].astype('category')
    data[categorical] = data[categorical].apply(lambda x: x.cat.codes)
    return data.to_numpy()


def kl_threshold(p, alpha):
    return p / (p + alpha * (1-p))


def kl_cost(span, p=0.5, alpha=1):
    return - p * np.log(span) - alpha * (1-p) * np.log(1-span)


def find_threshold(data, q=0.5):
    """Finds the threshold associated with the share of positives q in the dataset"""
    sorted = np.sort(data) 
    indices = np.flip(np.arange(start=0, stop=len(data))) # looking from right side
    mini = np.argmin(abs(indices/len(sorted)-q))
    return sorted[mini]


def exponential_mixture(span, p=0.5, l0=4, l1=6):
    f0 = np.exp(-l0 * span) * l0 / (1 - np.exp(-l0))
    f1 = - np.exp(l1 * span) * l1 / (1 - np.exp(l1))
    return p * f1 + (1-p) * f0, (1-p)*f0, p*f1


def exponential_threshold(p=0.5, alpha=1, l=5, epsilon=0.001):
    """Implementation of Newtons Method for finding q with exponential integral"""
    q = kl_threshold(p, alpha) 
    theta = 0
    theta_new = 0.5
    while abs(theta-theta_new) > epsilon:
        theta = theta_new
        f, d = exponential_integral(p, theta, l)
        theta_new = theta - (f-q)/d
    return theta


def exponential_integral(p=0.5, q=0.5, l=5):
    """Cost function for calculating I = p / (p + l * (1-p)) and its derivative"""
    f_val = np.exp(-l*q) * (np.exp(l)-np.exp(l*q)) * (p*(np.exp(l*q)-1) + 1) / (np.exp(l)-1)
    df_val = l * np.exp(-l*q) * (np.exp(l)*(p-1) - p*np.exp(2*l*q)) / (np.exp(l)-1)
    return f_val, df_val


def calculate_z_length(alpha=0.05, error=0.05, p=0.5, lamb=1, comp=False):
    """Calculates minimum amount of samples required for specified error in alpha CI"""
    sigma_q = np.sqrt(lamb) * (p*(1-p)) / (p + lamb*(1-p)) 
    z_alpha = norm.ppf(1-(1-alpha/2))
    comp = np.max([math.ceil(10/p), math.ceil(10/(1-p))]) if comp else 0
    return np.max([math.ceil((z_alpha/error)**2 * sigma_q**2), comp])