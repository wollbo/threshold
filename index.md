## Post model prediction threshold search

The binary prediction task is, perhaps, the most fundamental and important classification task in any classification framework. The quality of a given set of predictions is most aptly studied through ROC-analysis. However, for practical implementations it is necessary to select an operating threshold for a given set of predictions. 
There is no absolute best way or default framework for selecting this threshold, common practices include maximizing a given metric (F-1 score, Informedness) or reducing an overall expected cost induced by the erronous classification.
However, practical datasets are commonly unbalanced, and the costs associated with the different errors is dependent on the application. For example, when detecting a rare disease, the best classifier should perhaps be skewed towards predicting more False Positives than False Negatives, since an undetected disease might cause more harm than medicating a healthy person.
This indicates that a general framework for selecting the threshold should incorporate both information regarding class balance in the true dataset as well as the costs associated with the different errors.

### Minimizing distance between prediction distribution and true labels

One way to approach the problem of finding the best threshold is to minimize the average cost incurred from the false predictions, scaled with the cost associated with the respective errors. This has to be performed on a verification partition of our training data, however, since we should not have have access to the test dataset labels when selecting our threshold.
Another approach is to assume that the class balance in the training data is representative of the test data, such that we can assume that samples drawn from the test set follow the same distribution as that of the training set.
Through this assumption, we can posit that the distribution of predictions should also follow that of the data (since the optimal classifier has the same proportion of positives and negatives as the true data).
We can quantify this distance by measuring the KL-divergence between the class distribution of the gathered training data as well as that of the predictions for a given threshold.
In doing so, we can find an analytical, closed form solution for the threshold corresponding to a certain class balance. We can also introduce assymetrical weights associated with the different errors, leading to a solution on the form

```markdown
q = p / (p + lambda * (1-p))
```
{% include /custom/cost.html %}
