package project1.machinelearning

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object LogisticRegressionApp extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)
  val spark = SparkSession.builder().appName("LogisticRegressionApp")
    .master("local[*]")
    .config("spark.sql.warehouse.dir", "C:\\Users\\kanshu\\mydrive\\tmp")
    .getOrCreate()

  // Load and parse data
  val rdd = spark.sparkContext.textFile("data/wdbc.csv")
  def parseRDD(rdd: RDD[String]): RDD[Array[Double]] = {
    rdd.map(_.split(",")).filter(_(6) != "?").map(_.drop(1)).map(_.map(_.toDouble))
  }
  case class Cancer(cancer_class: Double, thickness: Double, size: Double, shape: Double,
                   madh: Double, epsize: Double, bnuc: Double, bchrom: Double, nNuc: Double,
                    mit: Double)
  def parseCancer(line: Array[Double]): Cancer = {
    Cancer( if (line(9) == 4.0) 1 else 0, line(0), line(1), line(2), line(3), line(4), line(5),
      line(6), line(7), line(8))
  }
   val cancerRdd = parseRDD(rdd).map(parseCancer)

  // Convert RDD to DataFrame for the ML pipeline
  import spark.sqlContext.implicits._
  val cancerDF = cancerRdd.toDF().cache()
//  cancerDF.show(false)

  // Feature extraction and transformation
  val featuresCols = Array("thickness", "size", "shape", "madh", "epsize", "bnuc",
    "bchrom", "nNuc", "mit")
  // assemble them into a feature vector
  val assembler = new VectorAssembler().setInputCols(featuresCols).setOutputCol("features")
  // transform them into a DataFrame
  val df2 = assembler.transform(cancerDF)
//  df2.show(false)
  // use the StringIndexer and create the label for the training dataset
  val labelIndexer = new StringIndexer()
    .setInputCol("cancer_class").setOutputCol("label")
  val df3 = labelIndexer.fit(df2).transform(df2)
//  df3.show(false)

  // Create test and training set
  val splitSeed = 123457
  val Array(trainingData , testData ) = df3.randomSplit(Array(0.7,0.3), splitSeed)

  // Creating an estimator using the training sets
  // create an estimator for the pipeline using the logistic regression
  val lr = new LogisticRegression().setMaxIter(50).setRegParam(0.01).setElasticNetParam(0.01)
  val model = lr.fit(trainingData)

  //Getting raw prediction, probability, and prediction for the test set
  val predictions = model.transform(testData)
  predictions.show()

  // Generating objective history of training
  val trainingSummary = model.summary
  val objectHistory = trainingSummary.objectiveHistory
//  objectHistory.foreach(println)

  // Evaluating the model
  val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]
  val roc = binarySummary.roc
  roc.show(false)
  println("Area under ROC curve" + binarySummary.areaUnderROC)

  /*
  Now let’s compute other metrics, such as true positive rate,
false positive rate, false negative rate, and total count, and a number of
instances correctly and wrongly predicted
   */
  import org.apache.spark.sql.functions._
  val lp = predictions.select("label", "prediction")
  val countTotal = predictions.count()
  val correct = lp.filter($"label" === $"prediction").count()
  val wrong = lp.filter(not($"label" === $"prediction")).count()
  val truep = lp.filter($"prediction" === 0.0).filter($"label" === $"prediction").count()
  val falseN = lp.filter($"prediction" === 0.0).filter(not($"label" === $"prediction")).count()
  val falseP = lp.filter($"prediction" === 1.0).filter(not($"label" === $"prediction")).count()
  val ratioWrong = wrong.toDouble / countTotal.toDouble
  val ratioCorrect = correct.toDouble / countTotal.toDouble

  println("Total Count: " + countTotal)
  println("Correctly Predicted: " + correct)
  println("Wrongly Identified: " + wrong)
  println("True Positive: " + truep)
  println("False Negative: " + falseN)
  println("False Positive: " + falseP)
  println("ratioWrong: " + ratioWrong)
  println("ratioCorrect: " + ratioCorrect)

  // judge the accuracy of the model
  val fMeasure = binarySummary.fMeasureByThreshold
  val fm = fMeasure.col("F-Measure")
  val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
  val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure).select("threshold").head().getDouble(0)
  model.setThreshold(bestThreshold)
  // compute the accuracy,
  val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")
  val accuracy = evaluator.evaluate(predictions)
  println("Accuracy: " + accuracy)
}
