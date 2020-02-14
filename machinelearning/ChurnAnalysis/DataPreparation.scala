package project1.machinelearning.ChurnAnalysis

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.{Encoders, Row}

object DataPreparation {

  case class CustomerAccount(state_code: String,
                             account_length: Int,
                             area_code: String,
                             international_plan: String,
                             voice_mail_plan: String,
                             num_voice_mail: Double,
                             total_day_mins: Double,
                             total_day_calls: Double,
                             total_day_charge: Double,
                             total_evening_mins: Double,
                             total_evening_calls: Double,
                             total_evening_charge: Double,
                             total_night_mins: Double,
                             total_night_calls: Double,
                             total_night_charge: Double,
                             total_international_mins: Double,
                             total_international_calls: Double,
                             total_international_charge: Double,
                             total_international_num_calls: Double,
                             churn: String
                            )
  val schema = Encoders.product[CustomerAccount].schema

  import spark.implicits._

  // let's see some related properties of the training set to understand its suitableness
  // Create a temp view for persistence for this session
  trainSet.createOrReplaceTempView("UserAccount")
  // create a catalog as an interface that can be used to create, drop, alter, or query underlying
  //databases, tables, functions, and many more
  spark.catalog.cacheTable("UserAccount")

  // Grouping the data by the churn label and calculating the number of instances in each group
  trainSet.groupBy("churn").count.show(false)

  // We can also observe that the preceding training set is highly unbalanced. Therefore, it
  //would be feasible to put two sample types on the same footing using stratified sampling
  // The sampleBy() method can be used to do so when provided with fractions of each sample type to be returned

  val fractions = Map("False" -> 0.1675, "True" -> 1.0)
  // This way, we are also mapping only True churn samples. Now, let's create a new DataFrame for the training
  // set containing only down sampled ones:

  val churnDF = trainSet.stat.sampleBy("churn", fractions, 12345L)

  churnDF.groupBy("churn").count.show(false)

  spark.sqlContext.sql("SELECT churn, SUM(total_day_charge) as TDC, SUM(total_evening_charge) as TEC," +
    "SUM(total_night_charge) as TNC, SUM(total_international_charge) as TIC, SUM(total_day_charge) + " +
    "SUM(total_evening_charge) + SUM(total_night_charge)  + SUM(total_international_charge) as Total_charge FROM " +
    "UserAccount GROUP BY churn ORDER BY Total_charge DESC").show(false)

  // let's see how many minutes of day, night, evening, and international voice calls have
  //contributed to the preceding total charge to the churn class

  spark.sqlContext.sql("SELECT churn, SUM(total_day_mins) + SUM(total_evening_mins) + " +
    "SUM(total_night_mins) + SUM(total_international_mins) as Total_minutes FROM UserAccount GROUP BY churn")
    .show(false)

  // Correlation calculation
  object Correlation {
    import org.apache.spark.mllib.linalg._

    val corTest = trainSet.select("num_voice_mail","total_day_mins","total_day_calls","total_day_charge"
      ,"total_evening_mins","total_evening_calls","total_evening_charge","total_night_mins","total_night_calls",
      "total_night_charge","total_international_mins","total_international_calls",
      "total_international_charge","total_international_num_calls").rdd
      .map{case Row(nvm: Double ,tdm: Double,tdc: Double,ttldc: Double,tem: Double,
      tec: Double,ttlec: Double,tnm: Double,tnc: Double,ttlnc: Double,tim: Double,tic: Double,ttlic: Double,tinc: Double) =>
        Vectors.dense(Array(nvm,tdm,tdc,ttldc,tem,tec,ttlec,tnm,tnc,ttlnc,tim,tic,ttlic,tinc))}

  }
/*
  import org.apache.spark.mllib.stat.Statistics
  val corMatrix = Statistics.corr(Correlation.corTest)

  println("correlation matrix")
  // println(corMatrix)

  // Convert the Matrix to RDD
  object MatrixToDF {
    import org.apache.spark.{SparkConf, SparkContext}
    import org.apache.spark.sql.{Row, SparkSession}
    import org.apache.spark.mllib.linalg._
    import org.apache.spark.rdd.RDD
    def toRDD(m: Matrix): RDD[Vector] = {
      val columns = m.toArray.grouped(m.numRows)
      val rows = columns.toSeq.transpose
      val vectors = rows.map(row => new DenseVector(row.toArray))
      spark.sparkContext.parallelize(vectors)
    }
  }
  val corrDF = MatrixToDF.toRDD(corMatrix).map(_.toArray)
    .map{case Array(nvm,tdm,tdc,ttldc,tem,tec,ttlec,tnm,tnc,ttlnc,tim,tic,ttlic,tinc) =>
      (nvm,tdm,tdc,ttldc,tem,tec,ttlec,tnm,tnc,ttlnc,tim,tic,ttlic,tinc)}
    .toDF("nvm","tdm","tdc","ttldc","tem","tec","ttlec","tnm","tnc","ttlnc","tim","tic","ttlic","tinc")
*/
  println("/n========================================================================/n")
  println("correlation matrix DF")
  // corrDF.show(false)

  // drop one column of each pair of correlated fields
  churnDF.select("account_length", "international_plan", "num_voice_mail",
    "total_day_calls","total_international_num_calls", "churn").show(10)

  /* The Spark ML API needs our data to be converted in a Spark DataFrame format,
  consisting of a label (in Double) and features (in Vector).
  we need to create a pipeline to pass the data through and chain several transformers and estimators.
  The pipeline then works as a feature extractor. More specifically, we have prepared two StringIndexer,
  transformers and a VectorAssembler.
  StringIndexer encodes a categorical column of labels to a column of label indices (that is, numerical).
  If the input column is numeric, we have to cast it into a string and index the string values.
  Other Spark pipeline components, such as Estimator or Transformer, make use of this stringindexed label.
  In order to do this, the input column of the component must be set to this string-indexed column name.
  In many cases, you can set the input column with setInputCol.
  */

  val ipindexer = new StringIndexer()
    .setInputCol("international_plan")
    .setOutputCol("iplanIndex")

  val labelindexer = new StringIndexer()
    .setInputCol("churn")
    .setOutputCol("label")

  val featureCols = Array("account_length", "iplanIndex", "num_voice_mail", "total_day_mins",
    "total_day_calls", "total_evening_mins", "total_evening_calls", "total_night_mins",
    "total_night_calls", "total_international_mins","total_international_calls", "total_international_num_calls")

  // let's transform the features into feature vectors, which are
  // vectors of numbers representing the value for each feature

  val assembler = new VectorAssembler()
    .setInputCols(featureCols)
    .setOutputCol("features")
  // Now that we have the real training set consisting of labels and feature vectors ready, the
  //next task is to create an estimator—the third element of a pipeline
  // Logistic Regression classifier
  // LR for churn prediction
  // LR is one of the most widely used classifiers to predict a binary response. It is a linear ML method


}
