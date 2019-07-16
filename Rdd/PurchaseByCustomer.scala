package com.kanshu.sparkscala.Rdd

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object PurchaseByCustomer extends App{

  Logger.getLogger("org").setLevel(Level.ERROR)

  def extractCustomerParis(line: String) = {
    val fields = line.split(",")
    (fields(0)toInt, fields(2).toFloat)
  }

  val sc = new SparkContext("local[*]", "PurchaseByCustomer")
  val input = sc.textFile("C:\\Users\\kanshu\\mydrive\\SparkScala\\SparkScala\\customer-orders.csv")

  val mappedInput = input.map(extractCustomerParis)
//  mappedInput.take(10).foreach(println)
  val totalByCustomer = mappedInput.reduceByKey((x,y)=> x + y)
  val results = totalByCustomer.map(x => (x._2, x._1)).sortByKey()
  results.collect().foreach(println)


}
