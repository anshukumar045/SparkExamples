package project1.machinelearning

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object MoviesRecommendationApp extends App{

  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder().appName("MoviesRecommendationApp")
    .master("local[*]")
    .config("spark.sql.warehouse.dir", "C:\\Users\\kanshu\\mydrive\\tmp")
    .getOrCreate()

  val ratingFile = "data/ratings.csv"
  val moviesFile = "data/movies.csv"

  val ratingDF = spark.read.format("csv").option("header", "true").load(ratingFile).cache()
//  ratingDF.show(false)

  val moviesDF = spark.read.format("csv").option("header", "true").load(moviesFile).cache()

  val numRatings = ratingDF.count()
  val numUsers = ratingDF.select("userId").distinct().count()
  val numMovies = ratingDF.select("movieId").distinct().count()
  println("got " + numRatings + " rating from "+ numUsers + " users on " + numMovies + " movies")

  ratingDF.createOrReplaceTempView("ratings")
  moviesDF.createOrReplaceTempView("movies")

  val results = spark.sql("select movies.title, movierates.maxr, movierates.minr, movierates.cntu "
  + "from(SELECT ratings.movieId,max(ratings.rating) as maxr,"
  + "min(ratings.rating) as minr,count(distinct userId) as cntu "
  + "FROM ratings group by ratings.movieId) movierates "
  + "join movies on movierates.movieId=movies.movieId "
  + "order by movierates.cntu desc")
//results.show(false)

  val mostActiveUser = spark.sql("SELECT ratings.userId, count(*) as ct from ratings " +
    "group by ratings.userId order by ct desc limit 10")
//  mostActiveUser.show(false)
  val result2 = spark.sql(
  "Select ratings.userId, ratings.movieId, ratings.rating, movies.title from ratings join movies " +
    " on movies.movieId = ratings.movieId " +
    "where ratings.userId = 414 and ratings.rating > 4")
//  result2.show(false)

  val splits = ratingDF.randomSplit(Array(0.75,0.25), seed = 12345L)
  val (trainingData, testData) = (splits(0), splits(1))
  val numTraining = trainingData.count()
  val numTest = testData.count()
  println("Training: " + numTraining + " test: " + numTest)

  // Prepare the data for building the recommendation model using ALS
  // The ALS algorithm takes the RDD of Rating for the training purpose

  val ratingRDD = trainingData.rdd.map( row => {
    val userId = row.getString(0)
    val movieId = row.getString(1)
    val ratings = row.getString(2)
    Rating(userId.toInt, movieId.toInt, ratings.toFloat)
  })
  import spark.implicits._
  val trData = ratingRDD.toDF("user", "item", "rating")

  val testRDD = testData.rdd.map( row => {
    val userId = row.getString(0)
    val movieId = row.getString(1)
    val ratings = row.getString(2)
    Rating(userId.toInt, movieId.toInt, ratings.toFloat)
  })
  /*
    val tstData = testRDD.toDF("user", "item", "rating")

    // Build an ALS user product matrix
    // this technique predicts missing ratings for specific users for specific movies based
    //on ratings for those movies from other users who did similar ratings for other movies



    val topRecsForUser = model.recommendForAllItems(414)
    val topRecsForUser1 = model.recommendForItemSubset(trData, 1)

  //  topRecsForUser1.show(false)

    val predictions = model.predictionCol
    println(predictions.toString())

  */
  val rank = 20
  val numIterations = 15
  val lambda = 0.10
  val alpha = 1.00
  val block = -1

  val seed = 12345L
  val implicitPrefs = false

//  val model = new ALS().setMaxIter(numIterations)
//    .setAlpha(alpha)
//    .setRank(rank)
//    .setSeed(seed)
//    .setImplicitPrefs(implicitPrefs)

//  val model = ALS.train(ratingRDD, rank, numIterations, block)

}
