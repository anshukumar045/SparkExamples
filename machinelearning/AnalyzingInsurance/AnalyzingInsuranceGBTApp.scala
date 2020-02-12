package project1.machinelearning.AnalyzingInsurance

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.regression.GBTRegressionModel
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.evaluation.RegressionMetrics

object AnalyzingInsuranceGBTApp extends App{

  Logger.getLogger("org").setLevel(Level.ERROR)

  import ModelPreparationGBT._
  import DataPreProcessing._
  import spark.implicits._
  // train the GBT Model

  println("training model with GradientBoostedTrees algorithm")
  val cvModel = cv.fit(trainingData)

  println("Evaluating model on train and test data and calculate RMSE")
  val trainPredictionsAndLabels =
    cvModel.transform(trainingData).select("label", "prediction")
    .map{ case Row(label: Double, prediction: Double ) => (label,prediction)}.rdd

  val validPredictionsAndLabels =
    cvModel.transform(validationData).select("label", "prediction")
    .map{case Row(label: Double, prediction: Double) => (label,prediction)}.rdd

  val trainRegressionMetrics = new RegressionMetrics(trainPredictionsAndLabels)
  val validRegressionMetrics = new RegressionMetrics(validPredictionsAndLabels)

  // Let's search for best model
  val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
  /*
  by using GBT it is possible to measure feature importance so that at a later stage we can decide which
  features are to be used and which ones are to be dropped from the DataFrame.
   */
  // Let's find the feature importance of the best model we just created previously,
  // for all features in ascending order as follows:

  val featureImportances = bestModel.stages.last.asInstanceOf[GBTRegressionModel].featureImportances.toArray

  val FI_to_List_sorted = featureImportances.toList.sorted.toArray

  val output =
    "\n=====================================================================\n" +
      s"Param trainSample: ${trainSample}\n" + s"Param testSample: ${testSample} \n" +
      s"TrainingData count: ${trainingData.count} \n" +
      s"ValidationData count: ${validationData.count} \n" +
      s"TestData count: ${testData.count}\n" +
      "=====================================================================\n" +
      s"Param maxIter = ${MaxIter.mkString(",")}\n" +
      s"Param maxDepth = ${MaxDepth.mkString(",")} \n" +
      s"Param numFolds = ${numFolds}\n" +
      "=====================================================================\n" +
      s"Training data MSE = ${trainRegressionMetrics.meanSquaredError}\n" +
      s"Training data RMSE = ${trainRegressionMetrics.rootMeanSquaredError}\n" +
      s"Training data R-squared = ${trainRegressionMetrics.r2} \n" +
      s"Training data MAE = ${trainRegressionMetrics.meanAbsoluteError} \n" +
      s"Training data Explained variance = ${trainRegressionMetrics.explainedVariance} \n" +
      "===================================================================== \n" +
      s"Validation data MSE = ${validRegressionMetrics.meanSquaredError}\n" +
      s"Validation data RMSE = ${validRegressionMetrics.rootMeanSquaredError} \n" +
      s"Validation data R-squared = ${validRegressionMetrics.r2}\n" +
      s"Validation data MAE = ${validRegressionMetrics.meanAbsoluteError}\n" +
      s"Validation data Explained variance = ${validRegressionMetrics.explainedVariance}\n" +
      "=====================================================================\n" +
      s"CV params explained: ${cvModel.explainParams}\n" +
      s"GBT params explained: ${bestModel.stages.last.asInstanceOf[GBTRegressionModel].explainParams}\n" +
      s"GBT features importances:n ${featureCols.zip(FI_to_List_sorted).map(t => s"t${t._1} = ${t._2}")
        .mkString("n")}n" +
      "=====================================================================\n"
  println(output)

  // let us run the prediction over the test set and generate the predicted loss for each claim
  // from the clients
  println("Run prediction over test dataset")
  cvModel.transform(testData)
    .select("id", "prediction")
    .withColumnRenamed("prediction", "loss")
    .coalesce(1)
    .write.format("csv")
    .option("header", "true")
    .save("output/result_GBT.csv")
}
