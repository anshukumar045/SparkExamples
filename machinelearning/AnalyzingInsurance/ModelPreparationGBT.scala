package project1.machinelearning.AnalyzingInsurance

object ModelPreparationGBT {
  // Implementing a GBT-based predictive model for predicting insurance severity claims
  /*
  In order to minimize a loss function, Gradient Boosting Trees (GBTs) iteratively train
  many decision trees. On each iteration, the algorithm uses the current ensemble to predict
  the label of each training instance.
  Then the raw predictions are compared with the true labels. Thus, in the next iteration, the
  decision tree will help correct previous mistakes if the dataset is re-labeled to put more
  emphasis on training instances with poor predictions.
   */

  // Strength of GBTs and its losses computation
  /**
   * ** N data instances
   * ** yi = label of instance i
   * ** xi = features of instance i
   * Then the F(xi) function is the model's predicted label; for instance, it tries to minimize the
   * error, that is, loss
   */
  /** Similar to decision trees, GBTs also
   * ** Handle categorical features (and of course numerical features too)
   * ** Extend to the multiclass classification setting
   * ** Perform both the binary classification and regression (multiclass classification is not yet supported)
   * ** Do not require feature scaling
   * ** Capture non-linearity and feature interactions, which are greatly missing in LR, such as linear models
    */
  // Validation while training: Gradient boosting can overfit, especially when you have trained your model
  // with more trees. In order to prevent this issue, it is useful to validate while carrying out the training

  import org.apache.spark.ml.regression.{GBTRegressor,GBTRegressionModel}
  import org.apache.spark.ml.{Pipeline, PipelineModel}
  import org.apache.spark.ml.evaluation.RegressionEvaluator
  import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}


  // let's define and initialize the hyperparameters needed to train the GBTs such as
  // the number of trees, number of max bins, number of folds to be used during cross-validation,
  // number of maximum iterations to iterate the training, and finally max tree depth

  val NumTrees = Seq(5,10,15)
  val MaxBins = Seq(28,28,28)
  val numFolds = 10
  val MaxIter: Seq[Int] = Seq(10)
  val MaxDepth: Seq[Int] = Seq(10)

  import spark.implicits._

  val modelGBT = new GBTRegressor()
    .setFeaturesCol("features")
    .setLabelCol("label")

  // build the pipeline by chaining the transformations and predictor together
  import DataPreProcessing._
  val pipeline = new Pipeline()
    .setStages((stringIndexerStages :+ assembler) :+ modelGBT)

  // Before crosvalidation set paramgrid
  val paramGrid = new ParamGridBuilder()
    .addGrid(modelGBT.maxIter, MaxIter)
    .addGrid(modelGBT.maxDepth, MaxDepth)
    .addGrid(modelGBT.maxBins, MaxBins)
    .build()

  println("preparing k- fold Cross Validation and Grid Search")

  val cv = new CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(new RegressionEvaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(numFolds)
}
