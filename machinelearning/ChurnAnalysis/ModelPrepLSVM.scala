package project1.machinelearning.ChurnAnalysis

object ModelPrepLSVM {
  // SVM is also used widely for large-scale classification that is,
  // (binary as well as multinomial)
  // The linear SVM algorithm outputs an SVM model, where the loss function
  //used by SVM can be defined using the hinge loss
  // The linear SVMs in Spark are trained with an L2 regularization, by default. However, it also
  //supports L1 regularization, by which the problem itself becomes a linear program

  import org.apache.spark.ml.classification.{LinearSVC, LinearSVCModel}
  import org.apache.spark.sql.functions.max
  import org.apache.spark.ml.Pipeline
  import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
  import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
  import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

  val numFolds = 10
  val MaxIter: Seq[Int] = Seq(100)
  val RegParam: Seq[Double] = Seq(1e-8) // L2 regularization param, set 0.10 with L1 regularization
  val Tol: Seq[Double] = Seq(1e-8)
  val ElasticNetParam: Seq[Double] = Seq(1.0) // Combination of L1 and L2

  // instantiate an LSVM estimator
  val svm = new LinearSVC()

  import DataPreparation._
  val pipeline = new Pipeline()
    .setStages(Array(ipindexer, labelindexer,assembler, svm))

  // define the paramGrid to perform such a grid search over the hyperparameter space
  // This searches through SVM's max iteration, regularization param, tolerance, and Elastic Net
  //for the best model

  val paramGrid = new ParamGridBuilder()
    .addGrid(svm.maxIter, MaxIter)
    .addGrid(svm.regParam, RegParam)
    .addGrid(svm.tol, Tol)
    .build()

  // define a BinaryClassificationEvaluator evaluator to evaluate the model
  val evaluator = new BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("prediction")

  // use a CrossValidator for performing 10-fold cross-validation for best model selection
  val crossval = new CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(numFolds)
}
