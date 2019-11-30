package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineModel, feature => mlFeature}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, GBTClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

object Trainer {

  def main(args: Array[String]): Unit = {

    //// ********************************************************************************
    //// (1) SET-UP
    //// ********************************************************************************
    // Turn off logs
    Logger.getLogger("org").setLevel(Level.ERROR)

    // Spark Session
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g",
      "spark.master" -> "local"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()

    val RANDOM_STATE: Long = 1492

    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline
      *       avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("Training Machine Learning Models on Kickstarter Kaggle Dataset.")


    //// ********************************************************************************
    //// (2) LOAD DATASET
    //// ********************************************************************************
    val PATH_TO_TRAIN : String = "./data/train_cleaned.parquet"
    val dfTrain : DataFrame = spark.read.option("header", "true").parquet(PATH_TO_TRAIN)


    //// ********************************************************************************
    //// (3) STRING TOKENIZATION
    //// ********************************************************************************
    val regexTokenizer = new mlFeature.RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")


    //// ********************************************************************************
    //// (4) REMOVE STOPWORDS
    //// ********************************************************************************
    val stopWordsRemover = new mlFeature.StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("tokensWithoutSW")


    //// ********************************************************************************
    //// (5): COMPUTE TERM FREQUENCY - INVERSE DOCUMENT FREQUENCY (TFIDF)
    //// ********************************************************************************
    val countVectorizer = new mlFeature.CountVectorizer()
      .setInputCol("tokensWithoutSW")
      .setOutputCol("tokensTF")
      .setVocabSize(20000)

    val idfVectorizer = new mlFeature.IDF()
      .setInputCol("tokensTF")
      .setOutputCol("tfidfFeatures")


    //// ********************************************************************************
    //// (6) ONE-HOT ENCODING OF CATEGORICAL VARIABLES
    //// ********************************************************************************
    //// String Indexer for Country Column
    val strIndexerCountry = new mlFeature.StringIndexer()
      .setInputCol("country2").setOutputCol("countryIndexed")
      .setHandleInvalid("skip")

    //// String Indexer for Country Column
    val strIndexerCurrency = new mlFeature.StringIndexer()
      .setInputCol("currency2").setOutputCol("currencyIndexed")
      .setHandleInvalid("skip")

    /// One-Hot Encoding of Country and Currency Features
    val oneHotEncoder = new mlFeature.OneHotEncoderEstimator()
      .setInputCols(Array("countryIndexed", "currencyIndexed"))
      .setOutputCols(Array("CAT_country", "CAT_currency"))


    //// ********************************************************************************
    //// (7) ASSEMBLE ALL FEATURES INTO 1 VECTOR
    //// ********************************************************************************
    //// Spark MLlib requires all of the features to be
    //// collected into 1 column who value is a Vector

    val featuresList : Array[String] = Array("tfidfFeatures",
      "days_campaign",
      "hours_prep",
      "goal",
      "CAT_country",
      "CAT_currency")

    val featureVectorizer = new mlFeature.VectorAssembler()
      .setInputCols(featuresList).setOutputCol("features")


    //// ********************************************************************************
    //// (8) LOGISTIC REGRESSION
    //// ********************************************************************************
    val logRegModel = new LogisticRegression()
      .setElasticNetParam(1.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)


    //// ********************************************************************************
    //// (9) CREATE A MACHINE LEARNING PIPELINE
    //// ********************************************************************************
    val pipelineSteps = Array(
      regexTokenizer,       // Step 1
      stopWordsRemover,     // Step 2
      countVectorizer,      // Step 3
      idfVectorizer,        // Step 4
      strIndexerCountry,    // Step 5
      strIndexerCurrency,   // Step 6
      oneHotEncoder,        // Step 7-8
      featureVectorizer,    // Step 9
      logRegModel           // Step 10
    )
    val logRegPipeModel = new Pipeline().setStages(pipelineSteps)


    //// ********************************************************************************
    //// (10) TRAIN / TEST SPLIT WITH STRATIFIED SAMPLING
    //// ********************************************************************************
    //// We would like to ensure that the identical target
    //// variable (final_status) distribution is reproduced
    //// in both Train and Testing sets

    val fractions = Map(0 -> 0.9, 1 -> 0.9)
    val train = dfTrain.stat.sampleBy("final_status",
      fractions=fractions,
      seed=RANDOM_STATE)
    val test = dfTrain.except(train)


    //// ********************************************************************************
    //// (11) TRAIN LOGISTIC REGRESSION
    //// ********************************************************************************
    //// Train Model
    println("\n*****************************************************")
    println("LOGISTIC REGRESSION")
    println("*****************************************************\n")
    val logRegFitted = logRegPipeModel.fit(train)
    val dfWithSimplePreds = logRegFitted.transform(test)

    //// Evaluation Metrics
    val f1Evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    val rocEvaluator = new BinaryClassificationEvaluator()
      .setLabelCol("final_status")
      .setRawPredictionCol("probability")
      .setMetricName("areaUnderROC")

    println("Results:")
    println("AUC Score : " + rocEvaluator.evaluate(dfWithSimplePreds))
    println("F1 Score  : " + f1Evaluator.evaluate(dfWithSimplePreds))
    println("Confusion matrix :")
    dfWithSimplePreds.groupBy("final_status", "predictions").count.show()


    //// ********************************************************************************
    //// (12) HYPERPARAMETER OPTIMIZATION
    //// ********************************************************************************

    println("\n*****************************************************")
    println("LOGISTIC REGRESSION WITH HYPERPARAMETER OPTIMIZATION")
    println("*****************************************************\n")

    // Train-Validation Grid Search
    val paramGrid = new ParamGridBuilder()
      .addGrid(logRegModel.regParam, Array(0.001, 0.01, 0.1, 1.0))
      .addGrid(countVectorizer.vocabSize, Array(5000, 10000, 20000))
      .build()

    val gridSearchValidator = new TrainValidationSplit()
      .setSeed(RANDOM_STATE)
      .setEstimator(logRegPipeModel)
      .setEvaluator(f1Evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    val gridSearchModel = gridSearchValidator.fit(train)
    val gridSearchPreds = gridSearchModel.transform(test)

    println("Best Pipeline Hyperparameters :")
    val bestModel = gridSearchModel.bestModel.asInstanceOf[PipelineModel]
    val stages = bestModel.stages
    val countVectorizerStage = stages(2).asInstanceOf[mlFeature.CountVectorizerModel]
    val logRegStage = stages(8).asInstanceOf[LogisticRegressionModel]
    println("Optimal vocabSize (CountVectorizer) : " + countVectorizerStage.getVocabSize)
    println("Optimal regParam  (logRegModel)     : " + logRegStage.getRegParam)

    println("\nResults (Grid Search) :")
    println("AUC Score : " + rocEvaluator.evaluate(gridSearchPreds))
    println("F1 Score  : " + f1Evaluator.evaluate(gridSearchPreds))
    println("Confusion matrix :")
    gridSearchPreds.groupBy("final_status", "predictions").count.show()


    //// ********************************************************************************
    //// (13) SAVE MODEL
    //// ********************************************************************************
    println("Saving Trained Pipeline with Hyperparameter Search")
    val PATH_TO_RESULTS_FOLDER : String = "././ModelCache"
    bestModel.write.overwrite.save(PATH_TO_RESULTS_FOLDER)

    println("Model saved.")

  }
}