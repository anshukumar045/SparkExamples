package project1.machinelearning.ChurnAnalysis

object ModelPreparationRF {
  /* Random Forest is an ensemble technique that takes a subset of observations and a subset of
  variables to build decision trees—that is, an ensemble of DTs. More technically, it builds
  several decision trees and integrates them together to get a more accurate and stable prediction
   */

  import org.apache.spark.sql.functions._
  import org.apache.spark.sql.types._
  import org.apache.spark.sql._
  import org.apache.spark.ml.Pipeline
  import org.apache.spark.ml.classification.{RandomForestClassifier, RandomForestClassificationModel}
  import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
  import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
  import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}

  import DataPreparation._

  val numFolds = 10
  // Instantiate a DecisionTreeClassifier estimator
  val rf = new RandomForestClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setSeed(1234567L)

  // We have three transformers and an estimator ready, the next task is to chain in a
  // single pipeline—that is, each of them acts as a stage

  val pipeline = new Pipeline()
    .setStages(Array(ipindexer, labelindexer, assembler, rf))

  val paramGrid = new ParamGridBuilder()
    .addGrid(rf.maxDepth, 3 :: 5 :: 15 :: 20 :: 50 :: Nil )
    .addGrid(rf.featureSubsetStrategy, "auto" :: "all" :: Nil)
    .addGrid(rf.impurity, "gini" :: "entropy" :: Nil)
    .addGrid(rf.maxBins, 2 :: 5 :: 10 :: Nil)
    .addGrid(rf.numTrees, 10 :: 50 :: 100 :: Nil)
    .build()

  val evaluator = new BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("prediction")

  // Use a CrossValidator for performing 10-fold cross-validation for best model selection
  val crossval = new CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(numFolds)
}
