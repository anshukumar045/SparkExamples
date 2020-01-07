package project1.machinelearning

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.sql.SparkSession

object CountVectorizerApp extends App{

  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder().appName("CountVectorizerApp")
    .master("local[*]")
    .getOrCreate()

  val df = spark.createDataFrame(
    Seq((0, Array("jason", "David")),
      (1,Array("David","Martin")),
      (2, Array("Martin", "Jason")),
      (3, Array("Jason", "Daniel")),
      (4, Array("Daniel", "Marting")),
      (5,Array("Moahmed", "Jason")),
      (6, Array("David", "David")),
      (7, Array("Jason", "Martin")))).toDF("id", "name")

  val df1 = spark.createDataFrame(
    Seq((0, Array("jason"), "David"),
      (1,Array("David"),"Martin"),
      (2, Array("Martin"), "Jason"),
      (3, Array("Jason"), "Daniel"),
      (4, Array("Daniel"), "Marting"),
      (5, Array("Moahmed"), "Jason"),
      (6, Array("David"), "David"),
      (7, Array("Jason"), "Martin"))).toDF("id", "fname", "lname")

//  df.show(false)
  df1.show(false)

  // fit the CountVectorizerModel

  val cvModel: CountVectorizerModel = new CountVectorizer()
    .setInputCol("name")
    .setOutputCol("features")
    .setVocabSize(3)
    .setMinDF(2)
    .fit(df)

  val cvModel1: CountVectorizerModel = new CountVectorizer()
    .setInputCol("fname") // fname should be an Array
    .setOutputCol("features")
    .setVocabSize(3)
    .setMinDF(2)
    .fit(df1)

  // downstream the vectorizer using the extractor

  val featrues = cvModel.transform(df)
  val features1 = cvModel1.transform(df1)

  features1.show(false)
  featrues.show(false)

}
