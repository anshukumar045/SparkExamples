package project1.machinelearning

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession

object LinearRegressionApp extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder().appName("LinearRegressionApp")
    .master("local[*]")
    .config("spark.sql.warehouse.dir", "C:\\Users\\kanshu\\mydrive\\tmp")
    .getOrCreate()

  val data = MLUtils.loadLibSVMFile(spark.sparkContext, "data/mnist.bz2")
  val featureSize = data.first().features.size
  println("Feature Size:- " + featureSize)

  // split the data into training and test
  val splits = data.randomSplit(Array(0.75,0.25), seed = 12345L )
  val (training, test) = (splits(0), splits(1))

  // reduce the features using PCA
  val pca = new PCA(featureSize/2).fit(data.map(_.features))
  val training_pca = training.map(p => p.copy(features = pca.transform(p.features)))
  val test_pca = test.map(p => p.copy(features = pca.transform(p.features)))

  // iterate 20 times and train the LinearRegressionWithSGD
  val numIterations = 20
  val stepSize = 0.0001
  val model = LinearRegressionWithSGD.train(training, numIterations)
  val model_pca = LinearRegressionWithSGD.train(training_pca, numIterations)

  /*
  evaluate the classification model,first, let’s prepare for computing
the MSE for the normal to see the effects of dimensionality reduction on the
original predictions. Obviously, if you want a formal way to quantify the
accuracy of the model and potentially increase the precision and avoid
overfitting. Nevertheless, you can do from residual analysis. Also it would be
worth to analyse the selection of the training and test set to be used for the
model building and then the evaluation.
   */

  // Evaluating both models
  val valuesAndPreds = test.map { point =>
    val score = model.predict(point.features)
    (score, point.label)
  }

  val valuesAndPreds_pca = test_pca.map{ point =>
    val score = model_pca.predict(point.features)
    (score, point.label)
  }

  val MSE = valuesAndPreds.map { case (v, p) => math.pow(v - p, 2) }.mean()
  val MSE_pca = valuesAndPreds_pca.map { case (v, p) => math.pow(v - p, 2) }.mean()
  println("Mean Squared Error = " + MSE)
  println("PCA Mean Squared Error = " + MSE_pca)

  //Observing the model coefficient for both models
  println("Model coefficients:"+ model.toString())
  println("Model with PCA coefficients:"+ model_pca.toString())
}
