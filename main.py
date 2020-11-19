import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
import sys
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from data import core

pgf = False
dataset = "credit"
alphas = [1, 0.5, 2]

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
p = sum(y)/len(y)  # 70 % positive 30 % negative
x_train, x_test, y_train, y_test = train_test_split(x, y)

model = make_pipeline(StandardScaler(), LGBMClassifier())
model.fit(x_train, y_train)
pred = model.predict_proba(x_test)[:, 1]

#tpr, fpr, thres = metrics.roc_curve(y_test, pred)
#plt.plot(tpr, fpr)
#plt.ylabel('True positive rate')
#plt.xlabel('False positive rate')
#plt.show()
p = 0.2
l = 5
alpha = 2

span = np.linspace(start=0.0, stop=1.0, endpoint=True)
mix, f0, f1 = core.exponential_mixture(span, p=p, l1=l, l0=l)
plt.plot(span, mix)
plt.plot(span, f0)
plt.plot(span, f1)
plt.axvline(x=core.exponential_threshold(p, alpha=alpha, l=l))
print(1-core.kl_threshold(p, alpha))
plt.show()
sys.exit(0)

span = np.linspace(start=0.001, stop=1, endpoint=False)  # cost function not defined at 0, 1
plt.hist(pred[y_test == 1], density=True, bins=50)
plt.hist(pred[y_test == 0], density=True, bins=50)


for idx, a in enumerate(alphas):  # Basic plotting structure
    plt.axvline(x=core.kl_threshold(p, alpha=a), c=colors[idx])
plt.legend(labels=[f'$\lambda = {a}$' for a in alphas])
#plt.show()
plt.savefig(f'report/figures/thresholds_{dataset}.pgf')


#for idx, a in enumerate(alphas):  # Basic plotting structure
#    plt.plot(span, core.kl_cost(span, p, alpha=a), c=colors[idx])
#plt.legend(labels=[f'λ = {a}' for a in alphas])
#plt.show()


# training data predictions
# plt.hist(model.predict_proba(x_train)[:, 1][y_train == 0], bins=50)
# plt.hist(model.predict_proba(x_train)[:, 1][y_train == 1], bins=50)

