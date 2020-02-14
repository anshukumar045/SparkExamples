package project1.machinelearning.ChurnAnalysis

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.functions.not

object ChurnAnalysisRFApp extends App{
  Logger.getLogger("org").setLevel(Level.ERROR)

  // call the fit method so that the complete, predefined pipeline, including all
  // feature preprocessing and the DT classifier, is executed multiple times—each time with a
  // different hyperparameter vector:

  import DataPreparation._
  import ModelPreparationRF._

  val cvModel = crossval.fit(churnDF)

  // evaluate the predictive power of the DT model on the test dataset
  // transform the test set to the model pipeline, which will map the features
  // according to the same mechanism we described in the previous feature engineering step

  val predictions = cvModel.transform(testSet)
  predictions.show(10)

  // classification accuracy
  val accuracy = evaluator.evaluate(predictions)
  println("Classification accuracy: " + accuracy)

  // observe the area under the precision-recall curve and the area under the ROC curve based
  // on the following RDD containing the raw scores on the test set

  val predictionsAndLabels = predictions.select("prediction", "label")
    .rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))

  val metrics = new BinaryClassificationMetrics(predictionsAndLabels)

  println("Area under the precision-recall curve: "+ metrics.areaUnderPR)
  println("Area under the receiver operating characteristic (ROC) curve: " + metrics.areaUnderROC)

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

  /* Random Forest can be debugged to get the decision tree that was constructed during the
  classification.
   */
  /* we have 10-folds on CrossValidation and five-dimensional hyperparameter space
  cardinalities between 2 and 7.
  Now let's do some simple math: 10 * 7 * 5 * 2 * 3 * 6 = 12600 models!
  Note that we still make the hyperparameter space confined, with numTrees, maxBins, and
  maxDepth limited to 7. Also, remember that bigger trees will most likely perform better
   */
  /* From the preceding results, it can be seen that LR and SVM models have the same but
   higher false positive rate compared to Random Forest and DT. So we can say that DT and
   Random Forest have better accuracy overall in terms of true positive counts
   */
  /* Now, it's worth mentioning that using random forest, we are actually getting high accuracy,
  but it's a very resource, as well as time-consuming job; the training, especially, takes a
  considerably longer time as compared to LR and SVM. Therefore, if you don't have higher memory
  or computing power, it is recommended to increase the Java heap space prior to running
  this code to avoid OOM errors
*/
  // Finally, if you want to deploy the best model (that is, Random Forest in our case), it is
  // recommended to save the cross-validated model immediately after the fit() method
  // invocation

  cvModel.write.overwrite().save("model/RF_model_churn")
  /*
  Your trained model will be saved to that location. The directory will include:
  ** The best model
  ** Estimator
  ** Evaluator
  ** The metadata of the training itself
   */
  // restoring the same model
  val cvModelrestored = CrossValidatorModel.load("model/RF_model_churn")


}
