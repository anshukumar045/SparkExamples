package com.kanshu.sparkscala.Rdd

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import scala.io.Source
import java.nio.charset.CodingErrorAction
import scala.io.Codec
import scala.math.sqrt

object MoviesSimilarities extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)
  def loadMoviesNames(): Map[Int, String] = {
    implicit val codec = Codec("UTF-8")
    codec.onMalformedInput(CodingErrorAction.REPLACE)
    codec.onUnmappableCharacter(CodingErrorAction.REPLACE)

    var movieNames: Map[Int, String] = Map()
    val lines = Source.fromFile("C:\\Users\\kanshu\\mydrive\\SparkScala\\ml-100k\\u.item").getLines()
    for (line <- lines) {
      var fields = line.split('|')
      if (fields.length > 1) {
        movieNames += (fields(0).toInt -> fields(1))
      }
    }
    return movieNames
  }

  type MovieRating = (Int, Double)
  type UserRatingPair = (Int, (MovieRating,MovieRating))

  def makePairs(userRatings: UserRatingPair): ((Int,Int),(Double,Double)) = {
    val movieRating1 = userRatings._2._1
    val movieRating2 = userRatings._2._2

    val movie1 = movieRating1._1
    val rating1 = movieRating1._2
    val movie2 = movieRating2._1
    val rating2 = movieRating2._2
    ((movie1,movie2), (rating1,rating2))
  }

  def filterDuplicates(userRating: UserRatingPair): Boolean = {
    val movieRating1 = userRating._2._1
    val movieRating2 = userRating._2._2

    val movie1 = movieRating1._1
    val movie2 = movieRating2._1

    return movie1 < movie2
  }

  type RatingPair = (Double, Double)
  type RatingPairs = Iterable[RatingPair]

  def computeCosineSimilarities(ratingPairs: RatingPairs): (Double,Int) = {
    var numPairs: Int = 0
    var sum_xx: Double = 0.0
    var sum_yy: Double = 0.0
    var sum_xy: Double = 0.0

    for (pair <- ratingPairs) {
      val ratingX = pair._1
      val ratingY = pair._2

      sum_xx += ratingX * ratingX
      sum_yy += ratingY * ratingY
      sum_xy += ratingX * ratingX
      numPairs += 1
    }

    val numerator: Double = sum_xy
    val denominator = sqrt(sum_xx) * sqrt(sum_yy)

    var score: Double = 0.0
    if (denominator != 0) {
      score = numerator / denominator
    }
    return (score, numPairs)
  }


  val sc = new SparkContext("local[*]", "MovieSimilarities")
  println("\nLoading movie names")

  val nameDict = loadMoviesNames()
  val data = sc.textFile("C:\\Users\\kanshu\\mydrive\\SparkScala\\ml-100k\\u.data")
  // Map ratings to key / value pairs: userId => movie ID, rating
  val ratings = data.map(l => l.split("\t")).map(l => (l(0).toInt, (l(1).toInt, l(2).toDouble)))

  // Emit every movie rates together by the same user
  // Self-Join to find evey combination
  val joinRatings = ratings.join(ratings)
  
  // Filter out duplicate pairs
  val uniqueJoinedRatings = joinRatings.filter(filterDuplicates)
  val moviePairs = uniqueJoinedRatings.map(makePairs)
  val moviePairRatings = moviePairs.groupByKey()
  val moviePairSimilarities = moviePairRatings.mapValues(computeCosineSimilarities).cache()

  // Extract similarities for the movie we care about that are good
  if (args.length > 0) {
    val scoreThreshold = 0.97
    val coOccurenceThreshold = 50.0

    val movieId: Int = args(0).toInt
    
    // Filter for movies with this sim that are "good" as defined by
    // our quality thresholds above
    val filteredResults = moviePairSimilarities.filter ( x => {
      val pair = x._1
      val sim = x._2
      (pair._1 == movieId || pair._2 == movieId) && sim._1 > scoreThreshold && sim._2 > coOccurenceThreshold
    })

    // Sort by quality score
    val results = filteredResults.map(x => (x._2, x._1)).sortByKey(false).take(10)
    println("\nTop 10 similar movies for "+ nameDict(movieId))

    for (result <- results) {
      val sim = result._1
      val pair = result._2
      // Display the similarity result that isn't the movie we are looking at
      var similarMovieId = pair._1
      if (similarMovieId == movieId) {
        similarMovieId = pair._2
      }
      println(nameDict(similarMovieId) + "\tscore: "+ sim._1 + "\tstrength: "+ sim._2)
    }
  }

}
