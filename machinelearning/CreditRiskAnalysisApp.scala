package project1.machinelearning

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object CreditRiskAnalysisApp extends App{
  Logger.getLogger("org").setLevel(Level.ERROR)
  val spark = SparkSession.builder().appName("CreditRiskAnalysisApp")
    .master("local[*]")
    .config("spark.sql.warehouse.dir", "C:\\Users\\kanshu\\mydrive\\tmp")
    .getOrCreate()

  import spark.implicits._
  val creditRdd = parseRdd(spark.sparkContext.textFile("data/german.csv")).map(parseCredit)

  def parseRdd(rdd: RDD[String]): RDD[Array[Double]] = {
    rdd.map(line => line.split(","))
      .map{case Array(x,y@_*) => y.map(_.toDouble).toArray }
//      .map(_.map(_.toDouble))
  }

  def parseCredit(line: Array[Double]): Credit = {
    Credit(line(0), line(1) -1, line(3), line(4), line(5), line(6) - 1, line(7) -1, line(8), line(9) - 1,
    line(10) - 1, line(11) - 1, line(12) - 1, line(13), line(14) - 1, line(15) - 1,
    line(16) - 1, line(17) - 1, line(18) - 1, line(19) -1, line(20) -1 )
  }
  case class Credit (
                    credibility : Double, balance: Double, duration: Double, purpose: Double,
                    amount: Double, savings: Double, employment: Double, instPercent: Double,
                    sexMarried: Double, guarantors: Double, residenceDuration: Double, assets: Double,
                    age: Double, concCredit: Double, apartment: Double, credits: Double, occupation: Double,
                    dependents: Double, hasPhone: Double, foreign: Double
                    )
  import spark.sqlContext
//  import spark.sqlContext.implicits._
  val creditDF = creditRdd.toDF().cache()

  creditDF.show(false)
  creditDF.createOrReplaceTempView("credit")

  // Observing related statistics

  sqlContext.sql("SELECT credibility, avg(balance) as avgBalance, " +
    "avg(amount) as avgAmount, avg(duration) as avgDur from credit group by credibility" ).show(false)

  creditDF.describe("balance").show(false)
  creditDF.groupBy("credibility").avg("balance").show(false)

  // Feature vectors and labels creation
  // create the feature vector considering credibility
  val featureCols = Array("balance", "duration", "purpose", "amount", "savings", "employment", "instPercent", "sexMarried",
    "guarantors", "residenceDuration", "assets", "age", "concCredit","apartment", "credits", "occupation", "dependents", "hasPhone",
    "foreign")

  val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
  val df2 = assembler.transform(creditDF)

  df2.select("features").show()

  // new column as a label from the old response column creditability using StringIndexer
  val labelIndexer = new StringIndexer()
    .setInputCol("credibility").setOutputCol("label")

  val df3 = labelIndexer.fit(df2).transform(df2)
  df3.select("label", "features").show(false)

  // Prepare the training and test set
  val splitSeed = 5043
  val Array(trainingData, testData) = df3.randomSplit(Array(0.80, 0.20), splitSeed)

  //Train the random forest model
  val classifier = new RandomForestClassifier()
    .setImpurity("gini")
    .setMaxDepth(30)
    .setNumTrees(30)
    .setFeatureSubsetStrategy("auto")
    .setSeed(1234567)
    .setMaxBins(40)
    .setMinInfoGain(0.001)

  val model = classifier.fit(trainingData)

  // Compute the raw prediction for the test set
  val predictions = model.transform(testData)
  predictions.select("label", "rawPrediction", "probability", "prediction").show()


}
