package project1.machinelearning

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession

object RandomForestApp extends App{

  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder().appName("RandomForestApp")
    .master("local[*]")
    .config("spark.sql.warehouse.dir", "C:\\Users\\kanshu\\mydrive\\tmp")
    .getOrCreate()

  // load and parse Mnist data
  val data = MLUtils.loadLibSVMFile(spark.sparkContext,"data/mnist.bz2")
  val splits = data.randomSplit(Array(0.75,0.25), seed = 12345L)
  val training = splits(0).cache()
  val test = splits(1)

  // Run the training algorithm to build the model
  val numClasses = 10 // number of classes in MNIST data set
  val categoricalFeatuesInfo = Map[Int, Int]() // since all the features are numeric
  val numTrees = 50 // Use more in practice. More is better
  val featureSubsetStrategy = "auto" // Let the algorithm chose
  // supported values are auto, all, sqrt, log2 and onethird
  val impurity = "gini"
  // the impurity criteria is used only for the information gain calculation.
  // the supported values are gini and variance for classification and regression respectively
  val maxDepth = 30 // More is better in practice
  // max depth is maximum depth of the tree
  val maxBins = 32 // more is better in practice
  // maxbin signifies maximum number of bins used for splitting the features, where suggested value is 100
  // to get better results
  // Random seed is used for bootstrapping and choosing feature subset to avoid
  // the random nature of the result
  val model = RandomForest.trainClassifier(training, numClasses, categoricalFeatuesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
  // Compute the raw score on the test set
  val scoreAndLabels = test.map{ point =>
    val score = model.predict(point.features)
    (score, point.label)
  }

  // Instantiate a multiclass metrics for the evaluation
  val metrics = new MulticlassMetrics(scoreAndLabels)

  // Constructing the confusion matrix
  println("Confusion Matrix:")
  println(metrics.confusionMatrix)

  // Overall statistics
  val accuracy = metrics.accuracy
  println("Summary Statistics")
  println(s"Accuracy = $accuracy")

  // Precession by label
  val labels = metrics.labels
  labels.foreach{ l =>
    println(s"Precision ($l) = " + metrics.precision(l))
  }

  // Recall by label
  labels.foreach{ l =>
    println(s"Recall ($l) = " + metrics.recall(l))
  }

  // False positive rate by label
  labels.foreach{ l=>
    println(s"FPR($l) = " + metrics.falsePositiveRate(l))
  }

  // F-measure by label
  labels.foreach { l =>
    println(s"F1-Score ($l) = " + metrics.fMeasure(l))
  }

  // Compute overall statistics
  println(s"Weighted precision: ${metrics.weightedPrecision}")
  println(s"Weighted recall: ${metrics.weightedRecall}")
  println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
  println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")
  println(s"Accuracy: ${metrics.accuracy}")

}
