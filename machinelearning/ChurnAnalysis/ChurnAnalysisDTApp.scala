package project1.machinelearning.ChurnAnalysis

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.functions.not

object ChurnAnalysisDTApp extends App{
  Logger.getLogger("org").setLevel(Level.ERROR)
  import DataPreparation._
  import ModelPreparationDT._

  // Call the fit method so that the complete predefined pipeline, including all
  // feature preprocessing and the DT classifier, is executed multiple times—each time with a
  // different hyperparameter vector

  val cvModel = crossval.fit(churnDF)

  // evaluate the predictive power of the DT model on the test dataset.
  // As a first step, we need to transform the test set with the model pipeline,
  // which will map the features according to the same mechanism we described in
  // the previous feature engineering step

  val predictions = cvModel.transform(testSet)
  predictions.show(10)

  // However, seeing the preceding prediction DataFrame, it is really difficult to guess the
  //classification accuracy. In the second step, in the evaluation is the evaluate itself using
  //BinaryClassificationEvaluator, as follows:
  val accuracy = evaluator.evaluate(predictions)
  println("Classification accuracy: " + accuracy)

  // observe the area under the precision-recall curve and the area under the ROC curve
  // based on the following RDD containing the raw scores on the test set

  val predictionsAndLabels = predictions.select("prediction", "label")
    .rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))

  // preceding RDD can be used to compute  aread under PR & ROC
  val metrics = new BinaryClassificationMetrics(predictionsAndLabels)
  println("Area under the precision-recall curve: "+ metrics.areaUnderPR)
  println("Area under Receiver operating characteristics: " + metrics.areaUnderROC())

  // false and true positive and negative predictions
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

  /*
  Fantastic; we achieved 87% accuracy, but for what factors? Well, it can be debugged to get
  the decision tree constructed during the classification. But first, let's see at what level we
  achieved the best model after the cross-validation
   */

  val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
  println("The Best Model and Parameters:n--------------------")
  println(bestModel.stages(3))

  // That means we achieved the best tree model at depth 5 having 43 nodes. Now let's extract
  // those moves (that is, decisions) taken during tree construction by showing the tree. This tree
  // helps us to find the most valuable features in our dataset

  bestModel.extractParamMap
  val treeModel = bestModel.stages(3).asInstanceOf[DecisionTreeClassificationModel]
  println("Learned classification tree model:n " + treeModel.toDebugString)

  // the toDebugString() function prints the tree's decision nodes
  //and the final prediction comes out at the end leaves


}
