package project1.machinelearning.ChurnAnalysis

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

object ModelPreparationLR {

  import DataPreparation._
  val numFolds = 10
  val MaxIter: Seq[Int] = Seq(100)
  val RegParam: Seq[Double] = Seq(1.0) // L2 regularization param, set 1.0 with L1 regularization
  val Tol: Seq[Double] =  Seq(1e-8) // For convergence tolerance for iterative algorithms
  val ElasticNetParam: Seq[Double] = Seq(0.0001) // Combination of L1 & L2
  // The RegParam is a scalar that helps adjust the strength of the constraints: a small value implies a soft margin,
  // so naturally, a large value implies a hard margin, and being an infinity is the hardest margin.

  // By default, LR performs an L2 regularization with the regularization parameter set to 1.0.
  // The same model performs an L1 regularized variant of LR with the regularization parameter
  // (that is, RegParam) set to 0.10. Elastic Net is a combination of L1 and L2 regularization.

  // On the other hand, the Tol parameter is used for the convergence tolerance for iterative
  // algorithms such as logistic regression or linear SVM
  //Once we have the hyperparameters defined and initialized, the next task is to instantiate an linear regression

  val lr = new LogisticRegression()
    .setLabelCol("label")
    .setFeaturesCol("features")

  // Now that we have three transformers and an estimator ready, the next task is to chain in a single
  // pipeline—that is, each of them acts as a stage

  val pipeline = new Pipeline()
    .setStages(Array(ipindexer, labelindexer , assembler, lr))

  val paramGrid = new ParamGridBuilder()
    .addGrid(lr.maxIter, MaxIter)
    .addGrid(lr.regParam, RegParam)
    .addGrid(lr.tol, Tol)
    .addGrid(lr.elasticNetParam, ElasticNetParam)
    .build()

  // Note that the hyperparameters form an n-dimensional space where n is the number of hyperparameters.
  // Every point in this space is one particular hyperparameter configuration, which is a hyperparameter vector
  // Of course, we can't explore every point in this space, so what we basically do
  //is a grid search over a (hopefully evenly distributed) subset in that space

  // Define a BinaryClassificationEvaluator evaluator, since this is a binary classification problem
  // Using this evaluator, the model will be evaluated according to a precision metric by comparing
  // the test label column with the test prediction column
  // The default metrics are an area under the precision-recall curve and an area under the
  // receiver operating characteristic (ROC) curve:

  val evaluator = new BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("prediction")

  // Use a CrossValidator for best model selection.
  // The CrossValidator uses the Estimator Pipeline, the Parameter Grid, and the Classification Evaluator

  /* The CrossValidator uses the ParamGridBuilder to iterate through the max iteration,
    regression param, and tolerance and Elastic Net parameters of linear regression, and then
    evaluates the models, repeating 10 times per parameter value for reliable results—that is,
    10-fold cross-validation
   */

  val crossval = new CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(numFolds)

  /* The preceding code is meant to perform cross-validation. The validator itself uses
    the BinaryClassificationEvaluator estimator for evaluating the training in the
    progressive grid space on each fold and makes sure that there's no overfitting.
    After calling fit, the complete predefined pipeline, including all feature preprocessing
    and the LR classifier, is executed multiple times—each time with a different hyperparameter vector
   */

  val cvModel = crossval.fit(churnDF)

}
