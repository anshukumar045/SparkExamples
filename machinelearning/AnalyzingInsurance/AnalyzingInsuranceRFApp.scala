package project1.machinelearning.AnalyzingInsurance

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.ml.regression. RandomForestRegressionModel
import org.apache.spark.ml.PipelineModel

object AnalyzingInsuranceRFApp extends App{
  Logger.getLogger("org").setLevel(Level.ERROR)

  import ModelPreparationRF._
  import DataPreProcessing._
  import spark.implicits._

  println("Training model with Random Forest algorithm")
  val cvModel = cv.fit(trainingData)

  // evaluating the model on the train and validation set, and calculating RMSE, MSE,
  //MAE, R-squared, and many more

  println("Evaluating model on train and validation set and calculating RMSE")

  val trainPredictionsAndLabels = cvModel.transform(trainingData)
    .select("label", "prediction")
    .map{ case Row(label: Double, prediction: Double) => (label,prediction)}.rdd

  val validPredictionsAndLabels = cvModel.transform(validationData)
    .select("label", "prediction")
    .map{ case Row(label: Double, prediction: Double) => (label,prediction)}.rdd

  val trainRegressionMetrics = new RegressionMetrics(trainPredictionsAndLabels)
  val validRegressionMetrics = new RegressionMetrics(validPredictionsAndLabels)
  /*
  using RF, it is possible to measure the feature importance so that at a
  later stage, we can decide which features should be used and which ones are to be dropped
  from the DataFrame. Let's find the feature importance from the best model we just created
  for all features in ascending order
   */
  val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
  val featureImportances = bestModel.stages.last.asInstanceOf[RandomForestRegressionModel]
    .featureImportances.toArray
  val FI_to_List_sorted = featureImportances.toList.sorted.toArray

  // observe the results on the train and the validation set

  val output =
    "\n=====================================================================\n" +
      s"Param trainSample: ${trainSample}\n" +
      s"TrainingData count: ${trainingData.count}\n" +
      s"ValidationData count: ${validationData.count}\n" +
      s"TestData count: ${testData.count}\n" +
      "=====================================================================\n" +
      s"Param maxIter = ${MaxIter.mkString(",")}\n" +
      s"Param maxDepth = ${MaxDepth.mkString(",")}\n" +
      s"Param numFolds = ${numFolds}\n" +
      "=====================================================================\n" +
      s"Training data MSE = ${trainRegressionMetrics.meanSquaredError}\n" +
      s"Training data RMSE = ${trainRegressionMetrics.rootMeanSquaredError}\n" +
      s"Training data R-squared = ${trainRegressionMetrics.r2}\n" +
      s"Training data MAE = ${trainRegressionMetrics.meanAbsoluteError}\n" +
      s"Training data Explained variance = ${trainRegressionMetrics.explainedVariance}\n" +
      "=====================================================================\n" +
      s"Validation data MSE = ${validRegressionMetrics.meanSquaredError}\n" +
      s"Validation data RMSE = ${validRegressionMetrics.rootMeanSquaredError}\n" +
      s"Validation data R-squared = ${validRegressionMetrics.r2}\n" +
      s"Validation data MAE = ${validRegressionMetrics.meanAbsoluteError}\n" +
      s"Validation data Explained variance =${validRegressionMetrics.explainedVariance}\n" +
      "=====================================================================\n" +
      s"CV params explained: ${cvModel.explainParams}\n" +
      s"RF params explained: ${bestModel.stages.last.asInstanceOf[RandomForestRegressionModel].explainParams}\n" +
      s"RF features importances:n ${featureCols.zip(FI_to_List_sorted).map(t => s"t${t._1} = ${t._2}").mkString("n")}n" +
      "=====================================================================\n"

  println(output)

}
