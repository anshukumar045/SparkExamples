package project1.Streaming

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.{Seconds, StreamingContext}

object StreamApp1 extends App{

  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder().appName("StreamApp1")
    .master("local[*]")
    .config("spark.sql.warehouse.dir", "C:\\Users\\kanshu\\mydrive\\tmp")
    .getOrCreate()

  val ssc = new StreamingContext(spark.sparkContext, Seconds(5))
  val fileStream = ssc.textFileStream("C:\\Users\\kanshu\\mydrive\\testdata\\stream")
  case class Toy (name: String, address: String)
//  fileStream.foreachRDD(rdd => {println(rdd.count())})
  import spark.implicits._
  fileStream.foreachRDD{ rdd =>
    val df = rdd.map(e => e.split(",")).toDF()
    println(df.show(1))
  }
     ssc.start()
 }
