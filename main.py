import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
import sys
from sklearn import metrics, model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from data import core


def arg_parse():
    """Parsing input arguments"""
    parser = argparse.ArgumentParser(description='KL-D optimal threshold selection')

    # Main experiment parameters
    parser.add_argument('-d', nargs=1, type=str, default="breastcancer", help="Dataset")
    parser.add_argument('-p', dest='plot', default=False, action='store_true', help="Saving plots")
    parser.add_argument('-l', nargs=1, type=list, default=[1, 2, 0.5], help="Evaluated FP/FN ratios")
    parser.add_argument('-e', nargs=1, type=int, default=[0], help="Exponential prediction")
    parser.add_argument('-r', dest='roc', default=False, action='store_true', help="Plot ROC curve")
    parser.add_argument('-c', dest='cost', default=False, action='store_true', help="Plot cost function")

    # ML-model parameters

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()

    dataset = args.d[0]  # dataset
    lambdas = args.l  # list of evaluated fp/fn ratios
    exp_pred = args.e[0]  # exponential test prediction parameter
    roc = args.roc  # plot roc curve of prediction
    pgf = args.plot
    cost = args.cost

    if pgf:
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    x, y = core.load_data(dataset)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)

    p = sum(y_train) / len(y_train)  # note: varying data structure between datasets

    model = make_pipeline(StandardScaler(), LGBMClassifier())
    model.fit(x_train, y_train)
    pred = model.predict_proba(x_test)[:, 1]  # Model outputs both class probabilities

    if roc:
        tpr, fpr, thres = metrics.roc_curve(y_test, pred)
        plt.plot(tpr, fpr)
        plt.ylabel('True positive rate')
        plt.xlabel('False positive rate')
        plt.show()

    if exp_pred:
        span = np.linspace(start=0.0, stop=1.0, endpoint=True)
        mix, f0, f1 = core.exponential_mixture(span, p=p, l1=exp_pred, l0=exp_pred)
        plt.plot(span, mix)
        plt.plot(span, f0)
        plt.plot(span, f1)
        for idx, a in enumerate(lambdas):
            plt.axvline(x=core.exponential_threshold(p, alpha=a, l=exp_pred), c=colors[-idx], lw=2)
        plt.legend(labels=[f'$\lambda = {a}$' for a in lambdas])
        print(f'p = {p}')
        print(1-core.kl_threshold(p, lambdas[0]))
        plt.savefig(f'report/figures/thresholds_{dataset}_exponential_{exp_pred}.pgf') if pgf else plt.show()
        plt.close(fig='all')

    span = np.linspace(start=0.001, stop=1, endpoint=False)  # cost function not defined at endpoints (0, 1)
    plt.hist(pred[y_test == 1], density=True, alpha=0.65, bins=50)
    plt.hist(pred[y_test == 0], density=True, alpha=0.65, bins=50)

    for idx, a in enumerate(lambdas):  # Basic plotting structure
        plt.axvline(x=core.kl_threshold(p, alpha=a), c=colors[-idx], lw=2)
    plt.legend(labels=[f'$\lambda = {a}$' for a in lambdas])
    plt.savefig(f'report/figures/thresholds_{dataset}.pgf') if pgf else plt.show()

    if cost:
        for idx, a in enumerate(lambdas):  # Basic plotting structure
            plt.plot(span, core.kl_cost(span, p, alpha=a), c=colors[-idx])
        plt.legend(labels=[f'λ = {a}' for a in lambdas])
        plt.show()


    # training data predictions
    # plt.hist(model.predict_proba(x_train)[:, 1][y_train == 0], bins=50)
    # plt.hist(model.predict_proba(x_train)[:, 1][y_train == 1], bins=50)

