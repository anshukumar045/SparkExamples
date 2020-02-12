package project1.machinelearning.AnalyzingInsurance

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.Row

object AnalyzingInsuranceLRApp extends App{
  Logger.getLogger("org").setLevel(Level.ERROR)

  import ModelPreparationLR._
  import DataPreProcessing._
  import spark.implicits._

  println("Training model with Linear Regression algorithm")
  val cvModel = cv.fit(trainingData)

  /* Now that we have the fitted model, that means it is now capable of making predictions.
  So let's start evaluating the model on the train and validation set and calculating RMSE, MSE,
  MAE, R-squared, and many more
   */

  println("Evaluating model on train and validation set and calculating RMSE")
  val trainPredictionsAndLabels =
    cvModel.transform(trainingData)
      .select("label", "prediction")
      .map{case Row(label: Double, prediction: Double) => (label,prediction)}.rdd

  val validPredictionsAndLabels =
    cvModel.transform(validationData)
      .select("label", "prediction")
      .map{case Row(label: Double, prediction: Double) => (label,prediction)}.rdd

  val trainRegressionMetrics = new RegressionMetrics(trainPredictionsAndLabels)
  val validRegressionMetrics = new RegressionMetrics(validPredictionsAndLabels)
  val bestModels = cvModel.bestModel.asInstanceOf[PipelineModel]

  // Once we have the best fitted and cross-validated model, we can expect good prediction accuracy.
  // let's observe the results on the train and the validation set

  val results = "\n=========================================================\n" +
    s"Param testSample: ${testSample} \n" +
    s"TrainingData count: ${trainingData.count} \n" +
    s"ValidationData count: ${validationData.count} \n" +
    s"TestData count: ${testData.count} \n" +
    "=====================================================================\n" +
    s"Param maxIter = ${MaxIter.mkString(",")}n" + s"Param numFolds = ${numFolds} \n" +
    "=====================================================================\n" +
    s"Training data MSE = ${trainRegressionMetrics.meanSquaredError} \n" +
    s"Training data RMSE = ${trainRegressionMetrics.rootMeanSquaredError} \n" +
    s"Training data R-squared = ${trainRegressionMetrics.r2} \n" +
    s"Training data MAE = ${trainRegressionMetrics.meanAbsoluteError} \n" +
    s"Training data Explained variance = ${trainRegressionMetrics.explainedVariance} \n" +
    "=====================================================================\n" +
    s"Validation data MSE = ${validRegressionMetrics.meanSquaredError} \n" +
    s"Validation data RMSE = ${validRegressionMetrics.rootMeanSquaredError} \n" +
    s"Validation data R-squared = ${validRegressionMetrics.r2} \n" +
    s"Validation data MAE = ${validRegressionMetrics.meanAbsoluteError} \n" +
    s"Validation data Explained variance = ${validRegressionMetrics.explainedVariance} \n" +
    s"CV params explained: ${cvModel.explainParams} \n" +
    s"LR params explained: ${bestModels.stages.last.asInstanceOf[LinearRegressionModel].explainParams} \n " +
    "=====================================================================\n"
  println(results)

  // Make a prediction on the test set
  println("Run prediction on the test set")
  cvModel.transform(testData)
    .select("id", "prediction")
    .withColumnRenamed("prediction", "loss")
    .coalesce(1) // to get the predictions in a single csv file
    .write.format("csv")
    .option("header", "true")
    .save("output/result_LR.csv")

}
