# Laboratory 3

This laboratory will be the introduction to classification. We will focus on data sets and classification using classifiers from the `sklearn` library. During this laboratory, we will learn how to prepare data for teaching and testing, calculate various metrics, and use the k-fold cross validation mechanism.

Below is an example of using the classifier from the `sklearn` library.

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create classifier
svc = SVC()
# Train classifier
svc.fit(X_train, y_train)
# Predict test data
y_pred = svc.predict(X_test)
# Calculate accuracy
score = accuracy_score(y_pred, y_test)
```

## Exercises

Please put solutions in the [`solution.py`](solution.py) file.

### Exercise 1 (2 pts)

- Load data from a file [`phoneme.csv`](data/phoneme.csv)
- Divide the set into the feature set (`X`) and the set of labels (` y`). Use the `prepare_data` function from the` utils.py` file
- Make a simple division of the set into teaching (`X_train`,` y_train`) and test (`X_test`, `y_test`). Maintain a 30% ratio for learning and 70% for testing [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- Learn the classifier [`k-NN`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier) on the training set
- Calculate the prediction for the learned classifier on the test set
- Display on the screen [`confusion_matrix`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) and [`accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)

### Exercise 2 (3 pts)

- Load data from a file [`balance.csv`](data/balance.csv)
- Divide the set into the feature set (`X`) and the set of labels (` y`). Use the `prepare_data` function from the` utils.py` file
- Make a k-fold cross validation for `k = 10` for the loaded data [`k-fold cross validation`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
- For each fold:
  - Learn the classifier [`k-NN`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier) on the training set
  - Calculate the prediction for the learned classifier on the test set
  - Put the accuracy on the list for the classification of the training set on the given foil.
- Display on the screen the average accuracy of the classification and its standard deviation.
- Try using other [`metrics`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)

### Exercise 3 (5 pts)

For 3 different classifiers and any 2 sets from the data directory:

- Load data from a file
- Divide the set into the feature set (`X`) and the set of labels (` y`). Use the `prepare_data` function from the` utils.py` file
- Make a k-fold cross validation for `k = 10` for the loaded data [`k-fold cross validation`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) (Exercise 2)
- Calculate the average classification accuracy, average learning time and average prediction time.
- Generate dependency plots using the `plot_results` function from the `utils.py` file:
  - average accuracy and average learning time
  - average accuracy and average prediction time
- Answer in the comment which of the tested classifiers obtains:
  - The best accuracy
  - The best time
  - Optimal accuracy and time
