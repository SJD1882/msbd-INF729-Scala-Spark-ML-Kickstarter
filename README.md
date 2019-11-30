## Predicting successful Kickstarter campaigns with Scala and Spark ML

### Summary

Project repository for Télécom Paris MS Big Data INF729 *Apache Spark with Scala* course. Project goal is to use Apache Spark, Spark ML and Scala for:

- Extracting continuous, categorical and text features from a Kaggle dataset of successful and failed Kickstarter campaigns
- Predicting the success (or not) of future Kickstarter projects with supervised Learning algorithms

Repository contains two Scala files:

- `Preprocessor`: Scala functions / script for preprocessing the Kickstarter dataset.
- `Trainer`: using Spark ML's Scala API for training a Logistic Regression Classifier on the preprocessed Kickstarter dataset. For Logistic Regression, hyperparameter optimization with grid search was used to find the most optimal hyperparameters (Regularization parameter and CountVectorizer's vocabulary size).

### Data

[Kaggle Dataset: Funding Successful Projects on Kickstarter](https://www.kaggle.com/codename007/funding-successful-projects)

### Implementation details

To implement both `Preprocessor` and `Trainer`, fork the repository. Then, run the following commands in a Bash terminal for submitting `Preprocessor` to spark-submit:

```bash
cd msbd-INF729-Scala-Spark-ML-Kickstarter
chmod +x build_and_submit.sh
./build_and_submit.sh Preprocessor
```

After preprocessing, the same should be done for `Trainer`:

```bash
./build_and_submit.sh Trainer
```

### Results

We ran two models: a Logistic Regression Classifier and another with hyperparameters selected through grid search.

| Model | `RegParam` | `VocabSize` | F1-Score | ROC AUC |
|---| ---| --- | --- | --- |
| `LogisticRegression` | 0.0 | 20000 | 0.650 | 0.671 |
| `LogisticRegression` | 0.001 | 20000 | 0.665 | 0.722 |

### Commentary

The main bonus addition to the Course Project guidelines was to use Stratified Sampling (common practice when classification datasets have unbalanced class distribution) during Train/Test splitting to ensure Training and Test Sets have the same class distribution. This was done with:

```scala
val fractions = Map(0 -> 0.9, 1 -> 0.9)
val train = dfTrain.stat.sampleBy("final_status", fractions=fractions,
				  seed=RANDOM_STATE)
val test = dfTrain.except(train)
```

I also created a Scala function for counting and displaying the number of null values per column in a Spark DataFrame:

```scala
  def displayNullValues(df: DataFrame): Unit = {
    /* Display null values in Spark DataFrame */
    println("NULL values ?")
    df.select(
      df.columns.map(
        c => F.sum(F.col(c).isNull.cast("int")).alias(c)
      ): _*
    ).show()
  }
```

An additional metric, ROC AUC Score, was also used to evaluate the soundness of our Logistic Regression's predictions to complement the F1-Score.

