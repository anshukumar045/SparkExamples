package project1.machinelearning.ChurnAnalysis

object ModelPreparationDT {
  /* DTs (Decision Tree) are commonly considered a supervised learning technique used for solving
    classification and regression tasks. Each branch in a DT represents a possible decision,
    occurrence, or reaction, in terms of statistical probability. Compared to naive Bayes,
    DTs are a far more robust classification technique. The reason is that at first, the DT
    splits the features into training and test sets. Then, it produces a good generalization
    to infer the predicted labels or classes. Most interestingly, a DT algorithm can handle
    both binary and multiclass classification problems.
    For instance, DTs learn from the admission data to approximate a sine curve with a set
    of if...else decision rules. The dataset contains the record of each student who applied
    for admission, say, to an American university. Each record contains the graduate record
    exam score, CGPA score, and the rank of the column. Now we will have to predict who is
    competent based on these three features (variables). DTs can be used to solve this kind
    of problem after training the DT model and pruning unwanted branches of the tree.
    In general, a deeper tree signifies more complex decision les and a better-fitted model.
    Therefore, the deeper the tree, the more complex the decision rules and the more fitted the
    model is.
   */

  import org.apache.spark.sql._
  import org.apache.spark.sql.functions._
  import org.apache.spark.sql.types._
  import org.apache.spark.ml.Pipeline
  import org.apache.spark.ml.classification.{DecisionTreeClassifier, DecisionTreeClassificationModel}
  import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
  import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
  import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}

  val numFolds = 10
  // instantiate a DecisionTreeClassifier estimator
  val dTree = new DecisionTreeClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setSeed(1234567L)

  // we have three transformers and an estimator ready, the next task is to chain in a single pipeline—that is,
  // each of them acts as a stage
  import DataPreparation._
  val pipeline = new Pipeline()
    .setStages(Array(ipindexer, labelindexer, assembler, dTree))

  // Let's define the paramgrid to perform such a grid search over the hyperparameter space.
  // This search is through DT's impurity, max bins, and max depth for the best model.
  // Maximum depth of the tree: depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.
  // On the other hand, the maximum number of bins is used for separate continuous features
  // and for choosing how to split on features at each node. More bins give higher granularity
  // In short, we search through decision tree's maxDepth and maxBins parameters for the best
  // model

  val paramGridBuilder = new ParamGridBuilder()
    .addGrid(dTree.impurity, "gini" :: "entropy" :: Nil)
    .addGrid(dTree.maxBins, 2 :: 5 :: 10 :: 15 :: 20 :: 25 :: 30 :: Nil)
    .addGrid(dTree.maxDepth, 5 :: 10 :: 15 :: 20 :: 25 :: 30 :: 30 :: Nil)
    .build()

  /* In the preceding code segment, we're creating a progressive paramgrid through sequence
    format. That means we are creating the grid space with different hyperparameter
    combinations. This will help us provide the best model that consists of the most optimal
    hyperparameters.
   */
  // Define a BinaryClassificationEvaluator evaluator to evaluate the model
  val evaluator = new BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("prediction")

  val crossval = new CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGridBuilder)
    .setNumFolds(numFolds)




}
