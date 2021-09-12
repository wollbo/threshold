import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn import metrics, model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from data import core


def arg_parse():
    """Parsing input arguments"""
    parser = argparse.ArgumentParser(description='KL-D optimal threshold selection')

    # Main experiment parameters
    parser.add_argument('-d', nargs=1, type=str, default="breastcancer", help="Dataset")
    parser.add_argument('-p', dest='plot', default=True, action='store_true', help="Saving plots")
    parser.add_argument('-l', nargs=1, type=list, default=[1, 10, 0.1], help="Evaluated FP/FN ratios")
    parser.add_argument('-e', nargs=1, type=int, default=[0], help="Exponential prediction")
    parser.add_argument('-r', dest='roc', default=False, action='store_true', help="Plot ROC curve")
    parser.add_argument('-c', dest='cost', default=False, action='store_true', help="Plot cost function")
    parser.add_argument('-s', nargs=1, type=int, default=None, help="Random seed for splitting train/test data")

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
    seed = args.s[0] if args.s else None

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
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=seed)

    p = sum(y_train) / len(y_train)  # note: varying data structure between datasets
    print(f'p = {p}')
    N = len(y_test)
    print(f'test samples N: {N}')

    model = make_pipeline(StandardScaler(), LGBMClassifier())
    model.fit(x_train, y_train)
    pred = model.predict_proba(x_test)[:, 1]  # Model outputs both class probabilities
    span = np.linspace(start=0.001, stop=1, endpoint=False)  # cost function not defined at endpoints (0, 1)

    if roc:
        tpr, fpr, thres = metrics.roc_curve(y_test, pred)
        plt.plot(tpr, fpr)
        plt.plot(fpr, fpr, c='black', dashes=[4, 2])
        plt.plot()
        plt.ylabel('True positive rate')
        plt.xlabel('False positive rate')
        plt.show()

    if exp_pred:
        thresholds = []
        span = np.arange(0, 1, step=0.001)
        mix, f0, f1 = core.exponential_mixture(span, p=p, l1=exp_pred, l0=exp_pred)
        plt.xlim(0, 1)
        plt.ylim(0, 3.5)
        plt.xlabel("$x$")
        plt.ylabel("$f(x)$")
        # plt.plot(span, f0)  # plotting additional densities clutters the image
        # plt.plot(span, f1)
        for idx, a in enumerate(lambdas):
            threshold = core.exponential_threshold(p, alpha=a, l=exp_pred)
            plt.axvline(x=threshold, c=colors[idx+2], lw=2)
            thresholds.append(threshold)
        plt.legend(labels=[f'$\lambda = {a}$, $\\theta = {np.around(t, 2)}$' for (a,t) in zip(lambdas, thresholds)])
        plt.plot(span, mix, c=colors[1])  # set y-span 0 - 3.5

        plt.savefig(f'report/figures/thresholds_{dataset}_exponential_{exp_pred}.pgf') if pgf else plt.show()
        plt.close(fig='all')

    plt.hist(pred[y_test == 1], density=True, alpha=0.65, bins=50)
    plt.hist(pred[y_test == 0], density=True, alpha=0.65, bins=50)
    thresholds = []
    for idx, a in enumerate(lambdas):  # Basic plotting structure
        q = core.kl_threshold(p, alpha=a)
        print(f'q is {q}')
        threshold = core.find_threshold(pred, q) # find for which threshold where the number of positives = q
        print(f'theta is {threshold}')
        plt.axvline(x=threshold, c=colors[idx+2], lw=2)
        thresholds.append(threshold)
    plt.legend(labels=[f'$\lambda = {a}$, $\\theta = {np.around(t, 2)}$' for (a,t) in zip(lambdas, thresholds)])
    plt.savefig(f'report/figures/thresholds_{dataset}.pgf') if pgf else plt.show()

    if cost:
        for idx, a in enumerate(lambdas):  # Basic plotting structure
            plt.plot(span, core.kl_cost(span, p, alpha=a), c=colors[idx+2])
        plt.legend(labels=[f'λ = {a}' for a in lambdas])
        plt.show()


