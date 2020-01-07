package project1.machinelearning

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, HashingTF, IDF, StopWordsRemover, Tokenizer}
import org.apache.spark.sql.SparkSession

object TFIDFApp extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)
  val spark = SparkSession.builder().appName("TFIDFApp")
    .master("local[*]")
    .getOrCreate()

  val sentenceData = spark.createDataFrame(Seq(
    (0, "Hi I heard about Spark"),
    (0, "I wish Java could use case classes"),
    (1, "Logistic regression models are meat")
  )).toDF("label", "sentence")

  val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
  val wordsData = tokenizer.transform(sentenceData)
  wordsData.show(false)
  val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")
  val removed = remover.transform(wordsData).cache()
  removed.show(false)

  /*
   if numFeatures is less than the actual number of distinct words/tokens in the DataFrame
   you are guaranteed to have an 'incorrect' frequency for at least 1 token
   (i.e. different tokens will hash to the same bucket).
   */

  val hashingTF = new HashingTF().setInputCol("filtered")
    .setOutputCol("rawFeatures").setNumFeatures(20)
  val featurizedData = hashingTF.transform(removed)
  featurizedData.show(false)

  val cvModel: CountVectorizerModel = new CountVectorizer()
    .setInputCol("filtered")
    .setOutputCol("features")
    .fit(removed)

  val features = cvModel.transform(removed)
  features.show(false)

  val idf = new IDF().setInputCol("rawFeatures").setOutputCol("featuresIDF").fit(featurizedData)
  val rescaleData = idf.transform(featurizedData)
  rescaleData.show(false)

  val idf1 = new IDF().setInputCol("features").setOutputCol("featuresIDF").fit(features)
  val rescaleData1 = idf1.transform(features)
  rescaleData1.show(false)



}
