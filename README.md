# threshold

## abstract

A novel way of finding the tradeoff between positive and negative predictions in a binary setting is defined and explored using information theoretic principles. Given a set of predicted probabilities, an estimation of the class balance and the respective costs associated with false positives and false negatives, we can find an optimal threshold by minimizing the distance between our prediction distribution and the estimated true class balance. Reference code as well as a [report](https://github.com/wollbo/threshold/blob/main/report/main.pdf) with derivations is maintained in this repository.

## background 

There are several scenarios in which we might wish to classify a score or probability as belonging to either of two categories. This is applicable in many different fields; medicine, radar, hypothesis testing and prediction among others. After the predictions or tests have been performed, the question of where to put the discriminatory threshold arises, at which point should we recognize a prediction as belonging to one category or the other? In the field of binary prediction, this is usually achieved by studying metrics related to the Receiver Operating Characteristics (ROC) and other metrics derived from the Precision/Recall matrix. However, while we can make some reasoning related to our specific problem, we do not arrive at an unbiased method through which we select our threshold, instead often relying on qualitative justification (it just seems right or gives the 'right' results). Instead, we would like to have an objective method through which we arrive at a threshold. This could be achieved by i.e. selecting the point on the ROC-curve closest to the top left corner, or the point on the curve corresponding to maximal informedness (Youden). These methods, however, presumes the selection of a set of metrics and do not take into account the class balance of the data or the relative costs associated with each type of misclassification.

## theory

Instead of studying the metrics of a validation set to select the threshold, we can utilize the structure and distribution of the raw test predictions to guide our choice. Since we know that a perfect predictor makes no mistakes, it follows that the distribution of predictions of such a predictor is identical to that of the true class labels. Even though we know that our predictors are not perfect, we can still use the same reasoning to define our notion of optimality and attain a method for finding the threshold.

We can quantify the degree of similarity between two distributions by looking at their Kullback-Leibler divergence, which can be thought of as a distance measure. Two identical distributions have a KL-divergence that is zero, and we posit that the best threshold is the one that minimizes the difference between the predictions and the class distribution of the true data. We do not know the class labels of the test set a priori, however, but we can still assume that the class balance in the training data and the test data is more or less the same.

Furthermore, in real world applications, there is always different costs associated with false positive errors and false negative errors (Type 1 and Type 2 errors). We can capture this relation by multiplying one of the two terms in the KL-divergence with a variable *λ*, which dictates the relative weight we ascribe to the different error types.

Defining the proportion of positive samples in the data as *p* and the proportion of thresholded positive samples in the test data as *q*, we find that in order to minimize the KL-divergence, the threshold should be selected such that *q* = *p*/(*p* + *λ*(1-*p*)).

## implementation

The theory derived is utilized to calculate thresholds for a number of different ML-dataset predictions in Python. This is achieved by first estimating *p* from the labels of the training data, and then after performing the test predictions, picking the threshold found at the index which corresponds to the resulting proportion *q* positive samples in the sorted set of test predictions.




