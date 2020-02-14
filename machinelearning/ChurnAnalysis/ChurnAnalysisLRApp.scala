package project1.machinelearning.ChurnAnalysis

import org.apache.log4j.{Level, Logger}
import org.apache.spark._
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.Dataset


object ChurnAnalysisLRApp extends App{

  Logger.getLogger("org").setLevel(Level.ERROR)

  // evaluate the predictive power of the LR model we created using the test dataset,
  // which has not been used for any training or cross-validation so far—that is, unseen
  // data to the model
  // As a first step, we need to transform the test set to the model pipeline,
  // which will map the features according to the same mechanism we described in the
  // preceding feature engineering step

  import DataPreparation._
  import ModelPreparationLR._

  val predictions = cvModel.transform(testSet)
  val result  = predictions.select("label", "prediction", "probability")
  val resultDF = result.withColumnRenamed("prediction", "Predicted_label")
  resultDF.show(10)

  // The prediction probabilities can also be very useful in ranking customers according to their
  //likeliness to imperfection
  // However, seeing the previous prediction DataFrame, it is really difficult to guess the
  //classification accuracy. In the second step, the evaluator evaluates itself using
  //BinaryClassificationEvaluator, as follows

  val accuracy = evaluator.evaluate(predictions)
  println("Classification accuracy: " + accuracy)

  // researchers often recommend other performance metrics, such as area under the
  //precision-recall curve and area under the ROC curve. However, for this we need to
  //construct an RDD containing the raw scores on the test set

  val predictionAndLabels = predictions
    .select("prediction", "label")
    .rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))

  val metrics = new BinaryClassificationMetrics(predictionAndLabels)
  println("Area under the precision-recall curve: " + metrics.areaUnderPR())
  println("Area under the receiver operating characteristic (ROC) curve: " + metrics.areaUnderROC())

  // False and True Positive and negative predictions are also useful to evaluate the model's performance
  /**
   * ** True positive: How often the model correctly predicted subscription canceling
   * ** False positive: How often the model incorrectly predicted subscription canceling
   * ** True negative: How often the model correctly predicted no canceling at all
   * ** False negative: How often the model incorrectly predicted no canceling
   */
  import spark.sqlContext.implicits._
  val lp = predictions.select("label", "prediction")
  val counttotal = predictions.count()
  val correct = lp.filter($"label" === $"prediction").count()

  val wrong = lp.filter(not($"label" === $"prediction")).count()
  val ratioWrong = wrong.toDouble / counttotal.toDouble
  val ratioCorrect =  correct.toDouble / counttotal.toDouble

  val truep = lp.filter($"prediction" === 0.0 )
    .filter($"label" === $"prediction").count() / counttotal.toDouble

  val truen = lp.filter($"prediction" === 1.0)
    .filter($"label" === $"prediction").count() / counttotal.toDouble

  val falsep = lp.filter($"prediction" === 1.0)
    .filter(not($"label" === $"prediction")).count() / counttotal.toDouble

  val falsen = lp.filter($"prediction" === 0.0)
    .filter(not($"label" === $"prediction")).count() / counttotal.toDouble

  println("Total count: " + counttotal)
  println("Correct: " + correct)
  println("Wrong: " + wrong)
  println("Ratio wrong: " + ratioWrong)
  println("Ratio correct: " + ratioCorrect)
  println("Ratio true positive : " + truep)
  println("Ratio false positive : " + falsep)
  println("Ratio true negative : " + truen)
  println("Ratio false negative : " + falsen)

}
