package project1.machinelearning.AnalyzingInsurance

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

object ModelPreparationLR {
  // That's all we need before we start training the regression models. First, we start training the
  //LR model and evaluate the performance.
  /*
  LR for predicting insurance severity claims
  As you have already seen, the loss to be predicted contains continuous values, that is, it will
  be a regression task. So in using regression analysis here, the goal is to predict a continuous
  target variable, whereas another area called classification predicts a label from a finite set.
  Logistic regression (LR) belongs to the family of regression algorithms. The goal of
  regression is to find relationships and dependencies between variables. It models the
  relationship between a continuous scalar dependent variable y (that is, label or target) and
  one or more (a D-dimensional vector) explanatory variable (also independent variables,
  input variables, features, observed data, observations, attributes, dimensions, and data
  points) denoted as x using a linear function:

  LR models the relationship between a dependent variable y, which involves a linear
  combination of interdependent variables xi. The letters A and B represent constants that
  describe the y axis intercept and the slope of the line respectively
  y = A+Bx
  The distance between any data points (measured) and the line (predicted) is called the regression error.
  Smaller errors contribute to more accurate results in predicting unknown values.
  When the errors are reduced to their smallest levels possible, the line of best fit is created for the
  final regression error.
  Note that there are no single metrics in terms of regression errors; there are several as follows:
  ** Mean Squared Error (MSE): It is a measure of how close a fitted line is to data points.
    The smaller the MSE, the closer the fit is to the data.
  ** Root Mean Squared Error (RMSE): It is the square root of the MSE but probably the most easily
    interpreted statistic, since it has the same units as the quantity plotted on the vertical axis
  ** R-squared: R-squared is a statistical measure of how close the data is to the fitted regression line.
    R-squared is always between 0 and 100%. The higher the Rsquared, the better the model fits your data.
  ** Mean Absolute Error (MAE): MAE measures the average magnitude of the errors in a set of
    predictions without considering their direction. It's the average over the test sample of the
    absolute differences betw  on where all individual differences
    have equal weight.
  ** Explained variance: In statistics, explained variation measures the proportion to which a
    mathematical model accounts for the variation of a given dataset
   */

  // Developing insurance severity claims predictive model using LR
  // we will develop a predictive analytics model for predicting accidental loss against the severity
  // claim by clients

  val numFolds = 10
  val MaxIter: Seq[Int] = Seq(20)
  val RegParam: Seq[Double] = Seq(0.001)
  val Tol: Seq[Double] = Seq(1e-6)
  val ElasticNetParam: Seq[Double] = Seq(0.001)

  val modelLR = new LinearRegression()
    .setFeaturesCol("features")
    .setLabelCol("label")

  // Building Pipeline
  println("Building ML pipeline")
  import DataPreProcessing._

  val pipeline = new Pipeline()
    .setStages((stringIndexerStages :+ assembler) :+ modelLR)

  /**
   * Spark ML pipelines have the following components:
   * * DataFrame: Used as the central data store where all the original data and intermediate
   *    results are stored.
   * * Transformer: A transformer transforms one DataFrame into another by adding additional feature columns
   *    Transformers are stateless, meaning that they don't have any internal memory and
   *    behave exactly the same each time they are used.
   * * Estimator: An estimator is some sort of ML model. In contrast to a transformer, an estimator contains an internal state
   *    representation and is highly dependent on the history of the data that it has already seen
   * * Pipeline: Chains the preceding components, DataFrame, Transformer, and Estimator together.
   * * Parameter: ML algorithms have many knobs to tweak. These are called hyperparameters,
   *    and the values learned by a ML algorithm to fit data are called parameters.
   */

  // Create the paramgrid by specifying the number of maximum iterations, the value of the
  // regression parameter, the value of tolerance, and Elastic network parameters as follows

  val paramGrid = new ParamGridBuilder()
    .addGrid(modelLR.maxIter, MaxIter)
    .addGrid(modelLR.regParam, RegParam)
    .addGrid(modelLR.tol, Tol)
    .addGrid(modelLR.elasticNetParam, ElasticNetParam)
    .build()

  // for a better and stable performance, let's prepare the K-fold cross-validation and grid
  //search as a part of model tuning . perform 10-fold cross-validation

  println("Preparing K-fold Cross Validation and Grid Search: Model tuning")
  val cv = new CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(new RegressionEvaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(numFolds)

}
