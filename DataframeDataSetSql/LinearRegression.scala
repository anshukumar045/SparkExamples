package com.kanshu.sparkscala.DataframeDataSetSql

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.sql._
import org.apache.log4j._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.optimization.SquaredL2Updater

object LinearRegression extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)
  val sc = new SparkContext("local[*]", "LinearRegression")

  val trainingLines = sc.textFile("C:\\Users\\kanshu\\mydrive\\SparkScala\\SparkScala\\regression.txt")
  val testLines = sc.textFile("C:\\Users\\kanshu\\mydrive\\SparkScala\\SparkScala\\regression.txt")
  //Convert input data to LabeledPoints for MLLib
  val trainingData = trainingLines.map(LabeledPoint.parse).cache()
  val testData = testLines.map(LabeledPoint.parse)

  // linear regression model
  val algorithm = new LinearRegressionWithSGD()
  algorithm.optimizer
    .setNumIterations(100)
    .setStepSize(1.0)
    .setUpdater(new SquaredL2Updater)
    .setRegParam(0.01)

  val model = algorithm.run(trainingData)

  //Predict values for our test features data using our linear model
  val predictions = model.predict(testData.map(_.features))

  // Zip in the "real" values so we can compare them
  val predictionAndLabel = predictions.zip(testData.map(_.label))

  //Print out the predicted and actual values for each point
  for (predictions <- predictionAndLabel) {
    println(predictions)
  }


}
