package project1.machinelearning.ChurnAnalysis

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.functions._

object ChurnAnalysisLSVMApp extends App{
  Logger.getLogger("org").setLevel(Level.ERROR)

  import DataPreparation._
  import ModelPrepLSVM._
  val cvModel = crossval.fit(churnDF)

  // evaluate the predictive power of the SVM model on the test dataset
  val predictions = cvModel.transform(testSet)
  predictions.show(10)

  // the evaluator evaluates itself using BinaryClassificationEvaluator
  val accuracy = evaluator.evaluate(predictions)
  println("Classification accuracy: " + accuracy)

  // researchers often recommend other performance metrics, such as area under the
  //precision-recall curve and area under the ROC curve

  val predictionsAndLabels = predictions.select("prediction", "label")
    .rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))

  // area under the precision-recall curve and area under the ROC curve
  val metrics = new BinaryClassificationMetrics(predictionsAndLabels)
  println("Area under the precision-recall curve: "+ metrics.areaUnderPR())
  println("Area under the Receiver Operating Characteristic: " + metrics.areaUnderROC())

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

  /* Yet, we have not received good accuracy using SVM. Moreover, there is no option to select
  the most suitable features, which would help us train our model with the most appropriate
  features.
   */
}
