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

The main bonus addition to the Course Project guidelines was to use Stratified Sampling (common practice when classification datasets have unbalanced class distribution) during Train/Test splitting to ensure Training and Test Sets have the same class distribution. This was done with:

```scala
val fractions = Map(0 -> 0.9, 1 -> 0.9)
val train = dfTrain.stat.sampleBy("final_status", fractions=fractions,
				  seed=RANDOM_STATE)
val test = dfTrain.except(train)
```

### Implementation details

To implement both `Preprocessor` and `Trainer`, fork the repository:

```bash
cd Test
```

Then, run the following commands in a Bash terminal for submitting `Preprocessor` to spark-submit:

```bash
cd Test
```

After preprocessing, the same should be done for `Trainer`:

```bash
cd Test
```

### Results

We ran two models: a Logistic Regression Classifier and another with hyperparameters selected through a grid search.

| `RegParam` | `VocabSize` | F1-Score | ROC AUC |
| ---| --- | --- | --- |
| 0.0 | 20000 | 0.650 | 0.671 |
| 0.001 | 20000 | 0.665 | 0.722 |

