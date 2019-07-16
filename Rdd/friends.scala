package com.kanshu.sparkscala.Rdd

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object friends extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)

  val sc = new SparkContext("local[*]", "friends")

  def parseLines(line: String) = {
    val fields = line.split(",")
    val age = fields(2).toInt
    val name = fields(1).toString
    val numFriends = fields(3).toInt
    val extraFrd = numFriends - 50
    (name,age,numFriends, extraFrd)
  }

  val lines = sc.textFile("C:\\Users\\kanshu\\mydrive\\SparkScala\\SparkScala\\fakefriends.csv")
  val rdd = lines.map(parseLines)
//  rdd.mapValues(x =>(x,1)).collect().foreach(println)
//  rdd.map(x => (x._1,(x._2,1))).collect().foreach(println)
//  val totalByAge = rdd.mapValues(x =>(x,1)).reduceByKey((x,y)=> (x._1 + y._1, x._2 + y._2))
  val totalByAge = rdd.map(x => ((x._1,x._2),(x._3,x._4,1))).reduceByKey((x,y)=> (x._1 + y._1, x._2 + y._2,x._3 + y._3 ))
  totalByAge.collect().sorted.foreach(println)
  val averageByAge = totalByAge.mapValues(x => x._1 / x._2)
  val results = averageByAge.collect()
//  results.sorted.foreach(println)
}
