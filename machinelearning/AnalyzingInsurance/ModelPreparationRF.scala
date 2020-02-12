package project1.machinelearning.AnalyzingInsurance

object ModelPreparationRF {
  /* Random Forest is an ensemble learning technique used for solving supervised learning
  tasks, such as classification and regression. An advantageous feature of Random Forest is
  that it can overcome the overfitting problem across its training dataset. A forest in Random
  Forest usually consists of hundreds of thousands of trees. These trees are actually trained on
  different parts of the same training set.
  More technically, an individual tree that grows very deep tends to learn from highly
  unpredictable patterns.This creates overfitting problems on the training sets. Moreover,
  low biases make the classifier a low performer even if your dataset quality is good in terms
  of the features presented.
  On the other hand, an Random Forest helps to average multiple decision trees together with the
  goal of reducing the variance to ensure consistency by computing proximities between pairs of cases.
   */

  import org.apache.spark.ml.regression.RandomForestRegressor
  import org.apache.spark.ml.Pipeline
  import org.apache.spark.ml.evaluation.RegressionEvaluator
  import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}


  // define some hyperparameters
  // the number of folds for cross-validation, number of maximum iterations,
  // the value of regression parameters, value of tolerance
  // and elastic network parameters
  val NumTrees = Seq(5,10,15)
  val MaxBins = Seq(23,27,30)
  val numFolds = 10
  val MaxIter: Seq[Int] = Seq(20)
  val MaxDepth: Seq[Int] = Seq(20)

  /* an Random Forest based on a decision tree, we require maxBins to be at least
  as large as the number of values in each categorical feature. In our dataset, we have 110
  categorical features with 23 distinct values. Considering this, we have to set MaxBins to at
  least 23.
   */
  val modelRF = new RandomForestRegressor()
    .setFeaturesCol("features").setLabelCol("label")

  // build a pipeline estimator by chaining the transformer and the LR estimator
  println("Building ML pipeline")
  import DataPreProcessing._
  val pipeline = new Pipeline().setStages((stringIndexerStages :+ assembler) :+ modelRF)

  // set the Paramgrid
  val paramGrid = new ParamGridBuilder()
    .addGrid(modelRF.numTrees, NumTrees)
    .addGrid(modelRF.maxDepth, MaxDepth)
    .addGrid(modelRF.maxBins, MaxBins)
    .build

  // for better and stable performance, let's prepare the K-fold cross-validation and grid
  //search as a part of model tuning.

  println("Preparing K-fold Cross Validation and Grid Search: Model tuning")
  val cv = new CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(new RegressionEvaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(numFolds)

}
