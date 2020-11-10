import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
import sys
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from data import core

alphas = [1, 0.5, 2]

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

x, y = core.load_data("breastcancer")
p = sum(y)/len(y)  # 70 % positive 30 % negative
x_train, x_test, y_train, y_test = train_test_split(x, y)

model = make_pipeline(StandardScaler(), LGBMClassifier())
model.fit(x_train, y_train)
pred = model.predict_proba(x_test)[:, 1]

tpr, fpr, thres = metrics.roc_curve(y_test, pred)
plt.plot(tpr, fpr)
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.show()

span = np.linspace(start=0.001, stop=1, endpoint=False)  # cost function not defined at 0, 1
plt.hist(pred[y_test == 1], density=True, bins=50)
plt.hist(pred[y_test == 0], density=True, bins=50)


for idx, a in enumerate(alphas):  # Basic plotting structure
    plt.axvline(x=core.kl_threshold(p, alpha=a), c=colors[idx])
plt.legend(labels=[f'λ = {a}' for a in alphas])
plt.show()


for idx, a in enumerate(alphas):  # Basic plotting structure
    plt.plot(span, core.kl_cost(span, p, alpha=a), c=colors[idx])
plt.legend(labels=[f'λ = {a}' for a in alphas])
plt.show()


# training data predictions
# plt.hist(model.predict_proba(x_train)[:, 1][y_train == 0], bins=50)
# plt.hist(model.predict_proba(x_train)[:, 1][y_train == 1], bins=50)

