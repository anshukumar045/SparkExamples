package project1.machinelearning

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession

object MultiClassClassificationApp extends App{
  Logger.getLogger("org").setLevel(Level.ERROR)
  val spark = SparkSession.builder().appName("MultiClassificationApp")
    .master("local[*]")
    .config("spark.sql.warehouse.dir", "C:\\Users\\kanshu\\mydrive\\tmp")
    .getOrCreate()

  // load and parse Mnist data
  val data = MLUtils.loadLibSVMFile(spark.sparkContext,"data/mnist.bz2")
  val splits = data.randomSplit(Array(0.75,0.25), seed = 12345L)
  val training = splits(0).cache()
  val test = splits(1)

  // Run the training algorithm to build the model
  val model = new LogisticRegressionWithLBFGS()
    .setNumClasses(15).setIntercept(true).setValidateData(true).run(training)
  // if you want algorithm to validate the training set before the model building
  // set the value using the setValidateData() method

  // clear the default threshold
  model.clearThreshold()

  // Compare raw scors on the test set
  val scoreAndLabels = test.map { point =>
    val score = model.predict(point.features)
    (score, point.label)
  }

  // Instantiate a multi-class metrics for the evaluation
  val metrics = new MulticlassMetrics(scoreAndLabels)

  //Construction the confusion matrix
  // in confusion matrix , each column represents the instances in a predict class
  // while each row represents the instance in an actual class (or vice versa)
  // the name stems from the fact that it makes it easy to see if the system is confusing two classes
  println("Confusion matrix:")
  println(metrics.confusionMatrix)

  // Overall Statistics
  // Compute the overall statistics to judge the performance of the model
  val accuracy = metrics.accuracy
  println("Summary Statistics")
  println(s"Accuracy = $accuracy")
  // precision by label
  val labels = metrics.labels
  labels.foreach { l =>
    println(s"Precision($l) = " + metrics.precision(l))
  }
  // Recall by label
  labels.foreach { l =>
    println(s"Recall($l) = " + metrics.recall(l))
  }
  //False positive rate by label
  labels.foreach{ l =>
    println(s"FPR($l) " + metrics.falsePositiveRate(l))
  }
  // F-Measure by label
  labels.foreach{ l =>
    println(s"F1-Score($l) " + metrics.fMeasure(l))
  }
  // the overall statistics say that the accuracy of the model is more than 92%
  // however we can improve the performance using a better algorithm such as random forest
  println(s"Weighted precision: ${metrics.weightedPrecision}")
  println(s"Weighted recall: ${metrics.weightedRecall}")
  println(s"Weighted F1 score ${metrics.weightedFMeasure}")
  println(s"Weighted false positive ${metrics.weightedFalsePositiveRate}")
}
