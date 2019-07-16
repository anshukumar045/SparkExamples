package com.kanshu.sparkscala.Rdd

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object MostPopularSuperhero extends App {

  // Function to extract the hero ID and number of connections from each line
  def countCoOccurences(line: String): (Int, Int) = {
    val elements = line.split("||s+")
    (elements(0).toInt, elements.length-1)
  }

  // Function to extract hero ID -> hero name tuples(or None in the case of failure)
  def parseNames(line: String): Option[(Int, String)] = {
    val fields = line.split('\"')
    if (fields.length > 1) {
      return Some(fields(0).trim().toInt, fields(1))
    } else {
      return None
    }
  }

  Logger.getLogger("org").setLevel(Level.ERROR)
  val sc = new SparkContext("local[*]", "MostPopularSuperhero")

  val names = sc.textFile("C:\\Users\\kanshu\\mydrive\\SparkScala\\SparkScala\\Marvel-names.txt")
  val namesRdd = names.flatMap(parseNames)

//  namesRdd.take(10).foreach(println)

  val lines = sc.textFile("C:\\Users\\kanshu\\mydrive\\SparkScala\\SparkScala\\Marvel-graph.txt")
  val pairings = lines.map(countCoOccurences)

  pairings.take(10).foreach(println)
  println("*********************")
  val totalFriendsByCharacter = pairings.reduceByKey((x,y)=> x + y)
  totalFriendsByCharacter.take(10).foreach(println)
  println("*********************")
  val flipped = totalFriendsByCharacter.map(x => (x._2, x._1))
  val mostPopular = flipped.max()
  println(mostPopular)
  println("*********************")
  val mostPopularName = namesRdd.lookup(mostPopular._2)(0)
  println(mostPopularName)
  println(s"$mostPopularName is the most popular superhero ${mostPopular._1} co-appearance")

}
