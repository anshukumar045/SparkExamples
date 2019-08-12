package com.kanshu.sparkscala

import breeze.linalg.max
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.sql.functions._

object ParseWithList extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder().appName("ParseWithList")
    .master("local[*]")
    .config("spark.sql.warehouse.dir", "C:\\Users\\kanshu\\mydrive\\tmp")
    .getOrCreate()

  val lines = spark.sparkContext.textFile("C:\\Users\\kanshu\\mydrive\\SparkScala\\SparkScala\\fakefriends.csv")
  val rdd = lines.map(_.split(",").toList)
  val plst = rdd.map{case List(x, y @_*) => People(x, y.map(e => e.toString).toList)}
  val pplst = plst.map(e => List(e.pId, e.id, e.rest(1).toInt * 2,  e.rest))
  val sqlContext = new SQLContext(spark.sparkContext)
  val df = sqlContext.createDataFrame(rdd.map{case List(x, y @_*) => People(x, y.map(e => e.toString).toList)})
  df.printSchema()
  val newDf = sqlContext.createDataFrame(pplst.map{case List(x,y,z, a @_*) =>
    PeopleWithID(x.toString, y.toString, z.toString.toInt, a.map(_.toString).toList )})
  newDf.printSchema()

  import spark.implicits._
  val ds = sqlContext.createDataset(pplst.map{case List(x,y,z, a @_*) =>
    PeopleWithID(x.toString, y.toString, z.toString.toInt, a.map(_.toString).toList )})
  println("****DS****")
  ds.printSchema()
  val PeopleIDs = ds.map(_.PeopleID )
  val PeopleAge = ds.map(e => (e.PeopleID,(e.age , 1))).rdd.reduceByKey((x,y) => (x._1 + y._1, x._2 + y._2)).toDS()
//  PeopleAge.take(10).foreach(println)
  val PeopleRest = ds.flatMap(e =>  e.rest).rdd.map(f => f.split(",").toList).filter(e => e(1).trim.toInt > 50).flatMap(e => e).toDS()
  PeopleRest.take(10).foreach(println)



//  val pAge = ds.filter($"age" > 100).take(3)
//  pAge.foreach(println)

//  newDf.createOrReplaceTempView("people")
//  val result = spark.sql("select * from people where age > 100")
//  result.take(3).foreach(println)


//  val filteredAge = ds.filter(e => e.age % 3 == 0 )
//  filteredAge.take(1).foreach(println)


//  ds.take(4).foreach(println)

}

case class People(id: String, rest: List[String]) {
  def pId : String = rest(0) + "-" + id

}
case class PeopleWithID(PeopleID: String, id: String, age: Int, rest: List[String])
