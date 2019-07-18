package com.kanshu.sparkscala.DataframeDataSetSql

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.sql._
import org.apache.log4j._

object SparkSQL extends App {

  case class Person(ID: Int, name: String, age: Int, numFriends: Int, secretKey: String)

  def skey(a: String, b: String): String ={
    a + "-" + b
  }
  def mapper(line: String) : Person = {
    val fields = line.split(",")
    val person: Person = Person(fields(0).toInt, fields(1), fields(2).toInt, fields(3).toInt,skey(fields(1),fields(0)) )
    return person
  }

  Logger.getLogger("org").setLevel(Level.ERROR)
  val spark = SparkSession.builder().appName("SparkSql")
    .master("local[*]")
    .config("spark.sql.warehouse.dir", "C:\\Users\\kanshu\\mydrive\\tmp")
    .getOrCreate()

  import spark.implicits._

  val lines = spark.sparkContext.textFile("C:\\Users\\kanshu\\mydrive\\SparkScala\\SparkScala\\fakefriends.csv")
  val people = lines.map(mapper).toDS().cache()

  // Infer the schema, and register the DataSet as a table.
  people.printSchema()
  people.createTempView("people")

  people.select("secretKey").filter(people("age") >=13 && people("age") < 19).show()
  // SQL can be run over DataFrames that have been registered as a table
  val teenagers = spark.sql("SELECT * FROM people WHERE age >= 13 AND age < 19")
  val results = teenagers.collect()
//  results.foreach(println)
  spark.stop()

}
