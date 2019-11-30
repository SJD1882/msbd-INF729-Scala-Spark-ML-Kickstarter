package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{SparkSession, SaveMode}
import org.apache.spark.sql.{functions => F}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.DataFrame


object Preprocessor {

  /*** Data Cleaning Utilities
    ***/

  def displayNullValues(df: DataFrame): Unit = {
    /* Display null values in Spark DataFrame */
    println("NULL values ?")
    df.select(
      df.columns.map(
        c => F.sum(F.col(c).isNull.cast("int")).alias(c)
      ): _*
    ).show()
  }

  def displayVarCount(df: DataFrame, column: String,
                      topk: Integer = 20): Unit = {
    /* Display values count for a column */
    df.groupBy(column).count.orderBy(F.desc("count")).show(topk)
  }

  def cleanCountry(country: String, currency: String): String = {
    /* Clean Kickstarter Kaggle Dataset Country Column */
    if (country == "False"){ return currency }
    else { return country }
  }

  def cleanCurrency(currency: String, country: String): String = {
    /* Clean Kickstarter Kaggle Dataset Currency Column */
    if (currency != null && currency.length != 3) { return null }
    // else if (currency != null && country == "USD") { return "USD" }
    else { return currency }
  }

  def cleanIntegerColumn(value: Integer): Integer = {
    if (value == null){ return -1}
    else {return value}
  }

  def cleanDoubleColumn(value: Double): Double = {
    if (value == null){ return -1.0}
    else {return value}
  }

  def cleanStringColumn(value: String): String = {
    if (value == null){ return "unknown"}
    else {return value}
  }

  /*** Feature Engineering Utilities
    ***/

  def getDaysCampaign(launched_at: Integer, deadline: Integer): Double = {
    /* Feature Engineering: Length of Kickstarter Campaign
     * If null value, return -1.0
     * */
    if ((launched_at == null) || (deadline == null)){ return -1.0 }
    else {
      val nbDays : Double = (deadline - launched_at) / 86400.0
      return nbDays
    }
  }

  def getHoursPreparation(created_at: Integer,
                          launched_at: Integer): Double = {
    /* Feature Engineering: Length of Kickstarter Campaign Preparation
     * If null value, return -1.0
     * */
    if ((created_at == null) || (launched_at == null)){ return -1.0 }
    else {
      val nbHours : Double = (launched_at - created_at) / 3600.0
      return Math.round(nbHours * 1000.0) / 1000.0
    }

  }

  /*** Main
    * ***/

  def main(args: Array[String]): Unit = {

    //// ********************************************************************************
    //// (1) SET-UP
    //// ********************************************************************************
    // Turn off logs
    Logger.getLogger("org").setLevel(Level.ERROR)

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.master" -> "local"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select,
      *       drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/


    //// ********************************************************************************
    //// (2) LOAD KICKSTARTER DATA
    //// ********************************************************************************
    println("Preprocessing Kickstarter Kaggle Dataset.")

    val PATH_TO_FILE_1 : String = "./data/train.csv"

    val df = spark.read
      .option("header", "true")
      .csv(PATH_TO_FILE_1);
    // .parquet(PATH_TO_FILE_1);

    // println("Nb. of rows: " + df.count()) - 111792
    // println("Nb. of cols: " + df.columns.length) - 14


    //// ********************************************************************************
    //// (3) BASIC DATA PREPROCESSING
    //// ********************************************************************************
    println("Casting columns goal, deadline, state_changed_at, created_at, launched_at and backers_count as Integer columns.")

    //// (3.1) Changing Column Types
    val dfCasted = df
      .withColumn("goal", F.col("goal").cast("Int"))
      .withColumn("deadline", F.col("deadline").cast("Int"))
      .withColumn("state_changed_at", F.col("state_changed_at").cast("Int"))
      .withColumn("created_at", F.col("created_at").cast("Int"))
      .withColumn("launched_at", F.col("launched_at").cast("Int"))
      .withColumn("backers_count", F.col("backers_count").cast("Int"))
      .withColumn("final_status", F.col("final_status").cast("Int"))

    // println(dfCasted.printSchema())

    // println(
    //     dfCasted.select("goal", "backers_count", "final_status")
    //             .describe()
    //             .show()
    //     )

    // Display null values per column
    displayNullValues(dfCasted)

    // Display values count for several columns
    // displayVarCount(dfCasted, "disable_communication")
    // displayVarCount(dfCasted, "country")
    // displayVarCount(dfCasted, "currency")
    // displayVarCount(dfCasted, "deadline")
    // displayVarCount(dfCasted, "state_changed_at")
    // displayVarCount(dfCasted, "backers_count")
    // displayVarCount(dfCasted, "final_status")

    // Display duplicates
    // dfCasted.select("deadline").dropDuplicates.show()

    // Display multiple values count
    // dfCasted.groupBy("country", "currency")
    //        .count
    //        .orderBy(F.desc("count"))
    //        .show(50)


    //// (3.2) Drop useless columns
    println("Dropping columns disable_communication, backers_count and state_changed_at.")

    val dfNoFutur : DataFrame = dfCasted
      .drop("disable_communication",
        "backers_count",
        "state_changed_at")


    //// (3.3) Clean Currency and Country Columns

    // dfNoFutur.filter(F.col("country") === "false")
    //         .groupBy("currency")
    //         .count
    //         .orderBy(F.desc("count"))
    //         .show(50)

    println("Cleaning country and currency columns.")

    val udf_cleanCountry = F.udf(cleanCountry _)
    val udf_cleanCurrency = F.udf(cleanCurrency _)

    val dfCountry = dfNoFutur
      .withColumn("country2" , udf_cleanCountry(F.col("country"), F.col("currency")))
      .withColumn("currency2", udf_cleanCurrency(F.col("currency"), F.col("currency")))
      .drop("country", "currency")

    // dfCountry.groupBy("country2", "currency2")
    //        .count.orderBy(F.desc("count")).show(50)


    //// ********************************************************************************
    //// (4) FEATURE ENGINEERING
    //// ********************************************************************************

    //// (4.1) Prepare Target Variable
    println("Keep only final_status column values at 0 or 1.")

    val targetVarToKeep : List[Integer] = List(0, 1)

    val dfTarget = dfCountry
      .filter(F.col("final_status").isin(targetVarToKeep: _*))

    // displayVarCount(dfTarget, "final_status") // 0: 12399, 1: 8510


    //// (4.2) Adding New Time Features
    println("Creating new feature: days_campaign.")
    println("Creating new feature: hours_prep.")
    println("Dropping columns created_at, launched_at and deadline.")

    val udf_getDaysCampaign = F.udf(getDaysCampaign _)
    val udf_getHoursPrep = F.udf(getHoursPreparation _)

    val dfWithTimeFeatures = dfTarget
      .withColumn("days_campaign", udf_getDaysCampaign(
        F.col("launched_at"), F.col("deadline")
      ))
      .withColumn("hours_prep", udf_getHoursPrep(
        F.col("created_at"), F.col("launched_at")
      ))
      .drop("created_at", "launched_at", "deadline")

    // displayVarCount(dfWithTimeFeatures, "days_campaign")
    // displayVarCount(dfWithTimeFeatures, "hours_prep")


    //// (4.3) Adding New Text Features
    println("Lowering string columns name, desc and keywords.")

    val dfLowerString = dfWithTimeFeatures
      .withColumn("name", F.lower(F.col("name")))
      .withColumn("desc", F.lower(F.col("desc")))
      .withColumn("keywords", F.lower(F.col("keywords")))

    // dfWithTextFeatures.select("name", "desc", "keywords").show(10)

    println("Creating new feature: text.")
    println("Dropping name, desc and keywords columns.")

    val dfWithTextFeatures = dfLowerString
      .withColumn("text",
        F.concat(F.col("name"),
          F.lit(" "),
          F.col("desc"),
          F.lit(" "),
          F.col("keywords"))
      )
      .drop("name", "desc", "keywords")

    // dfWithTextFeatures.select("text").show(20)


    //// (4.4) Dealing with missing values

    // Missing values for goal (1082) and currency2 (395)
    // displayNullValues(dfWithTextFeatures)

    // displayVarCount(dfWithTextFeatures, "goal", 200)

    println("Cleaning goal and currency2 columns.")
    val udf_cleanIntegerColumn = F.udf(cleanIntegerColumn _)
    val udf_cleanStringColumn = F.udf(cleanStringColumn _)

    val dfFinal = dfWithTextFeatures
      .withColumn("goal",      udf_cleanIntegerColumn(F.col("goal")))
      .withColumn("currency2", udf_cleanStringColumn(F.col("currency2")))
      .withColumn("text",      udf_cleanStringColumn(F.col("text")))

    // displayNullValues(dfFinal)

    // dfFinal.show(10)


    //// ********************************************************************************
    //// (5) SAVE PARQUET DATA TO FILE
    //// ********************************************************************************
    println("Write Preprocessed Dataset to Parquet Format.")
    val PATH_TO_DATA_FOLDER : String = "./data"
    dfFinal.write.mode(SaveMode.Overwrite)
      .parquet(PATH_TO_DATA_FOLDER + "/train_cleaned.parquet")

  }

}