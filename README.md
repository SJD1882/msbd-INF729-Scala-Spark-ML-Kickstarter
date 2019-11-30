## Predicting successful Kickstarter campaigns with Scala and Spark ML

### Summary

Project repository for Télécom Paris MS Big Data INF729 *Apache Spark with Scala* course. Project goal is use Apache Spark, Spark ML and Scala for:

- Extracting continuous, categorical and text features from a Kaggle dataset of successful and failed Kickstarter campaigns
- Predicting the success (or not) of future Kickstarter projects with supervised Learning algorithms

Repository contains two Scala files:

- `Preprocessor`: Scala functions and script for preprocessing the Kickstarter dataset.
- `Trainer`: using Spark ML's Scala API for training Logistic Regression, Random Forests and Multilayer Perceptron Classifier on the preprocessed Kickstarter dataset. For Logistic Regression, hyperparameter optimization with grid search was to find the most optimal hyperparameters (Regularization parameter and CountVectorizer's vocabulary size).

### Data

[Kaggle Dataset: Funding Successful Projects on Kickstarter](https://www.kaggle.com/codename007/funding-successful-projects)

### Commentary

(Insert Additional Features and Models)

### Implementation details

To implement both `Preprocessor` and `Trainer`, run the following commands in a Bash terminal:

(Insert Bash Commands To Execute)

### Results

We ran two models: a Logistic Regression Classifier and another with hyperparameters selected through a grid search.

| `RegParam` | `VocabSize` | F1-Score | ROC AUC |
| ---| --- | --- | --- |
| 0.0 | 20000 | 0.650 | 0.671 |
| 0.001 | 20000 | 0.665 | 0.722 |

