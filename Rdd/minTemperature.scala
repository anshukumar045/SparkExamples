package com.kanshu.sparkscala.Rdd

import org.apache.spark._
import org.apache.log4j._
import scala.math.min

object minTemperature extends App {

  def parseLine(line: String): (String, String, Double) = {
    var fields = line.split(",")
    val stationID = fields(0)
    val entryTyp = fields(2)
    val temperature = fields(3).toFloat * 0.1f * (9.0f / 5.0f) + 32.0f
    (stationID,entryTyp,temperature)
  }

  Logger.getLogger("org").setLevel(Level.ERROR)

  val sc = new SparkContext("local[*]","MinTemperature")
  val lines = sc.textFile("C:\\Users\\kanshu\\mydrive\\SparkScala\\SparkScala\\1800.csv")
  val parsedLines = lines.map(parseLine)
  val minTemps = parsedLines.filter(x => x._2 == "TMIN")
  val stationTemps = minTemps.map(x => (x._1,x._3.toFloat))
  val minTempByStation = stationTemps.reduceByKey((x,y) => min(x,y))
  val results = minTempByStation.collect()

  for (result <- results.sorted) {
    val station = result._1
    val temp = result._2
    val formattedTemp = f"$temp%.2f F"
    println(s"$station minimum temperature: $formattedTemp")
  }


}
