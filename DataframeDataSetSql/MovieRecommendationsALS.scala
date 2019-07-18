package com.kanshu.sparkscala.DataframeDataSetSql

object MovieRecommendationsALS extends App {

  import org.apache.spark._
  import org.apache.spark.SparkContext._
  import org.apache.log4j._
  import scala.io.Source
  import java.nio.charset.CodingErrorAction
  import scala.io.Codec
  import org.apache.spark.mllib.recommendation._

  def loadMovieNames(): Map[Int, String] = {
    // Handle character encoding issues
    implicit val codec = Codec("UTF-8")
    codec.onMalformedInput(CodingErrorAction.REPLACE)
    codec.onUnmappableCharacter(CodingErrorAction.REPLACE)

    // Create Map of Ints to String , and populate it from u.item
    var movieNames: Map[Int, String] = Map()
    val lines = Source.fromFile("C:\\Users\\kanshu\\mydrive\\SparkScala\\ml-100k\\u.item").getLines()
    for (line <- lines ) {
      val fields = line.split('|')
      if (fields.length > 1) {
        movieNames += (fields(0).toInt -> fields(1))
      }
    }
    movieNames
  }

  Logger.getLogger("org").setLevel(Level.ERROR)
  val sc = new SparkContext("local[*]", "MovieRecomendationALS")

  println("Loading Movie Names")
  val nameDict = loadMovieNames()

  val data = sc.textFile("C:\\Users\\kanshu\\mydrive\\SparkScala\\ml-100k\\u.data")
  val ratings = data.map(x => x.split('\t')).map( x => Rating(x(0).toInt, x(1).toInt, x(2).toDouble)).cache()

  // Building Recommendation model using Alternating least square
  println("training recommendation Model")
  val rank = 8
  val numIterations = 20
  val model = ALS.train(ratings, rank, numIterations)
//  val userId = args(0).toInt
  val userId = 0
  println("\nRatings for user ID "+ userId + ":")
  val userRatings = ratings.filter(x => x.user == userId)
  val myRatings = userRatings.collect()

  for (rating <- myRatings) {
    println(nameDict(rating.product.toInt) + ": " + rating.rating.toString)
  }

  println("\nTo 10 recommendations")
  val recommendationns = model.recommendProducts(userId, 10)

  for (recommendation <- recommendationns) {
    println(nameDict(recommendation.product.toInt) + " score " + recommendation.rating)
  }
}
