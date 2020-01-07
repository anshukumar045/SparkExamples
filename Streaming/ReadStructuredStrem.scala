package project1.Streaming

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{Encoders, SparkSession}
import org.apache.spark.sql.streaming.OutputMode
import org.apache.spark.sql.types.{StringType, StructField, StructType}

object ReadStreamApp extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder().appName("ReadStreamApp1")
    .master("local[*]")
    .config("spark.sql.warehouse.dir", "C:\\Users\\kanshu\\mydrive\\tmp")
    .getOrCreate()

  case class people(name: String, address: String)

  val encoderSchema = Encoders.product[people].schema
  val schema = StructType(
    Array(StructField("name", StringType),
      StructField("address", StringType)))

  val fileStreamDf = spark.readStream
    .option("header", "true")
    .option("sep", ",")
    .schema(encoderSchema)
    .csv("C:\\Users\\kanshu\\mydrive\\testdata\\stream")

  val df = fileStreamDf.groupBy("name").count()

  val query = df.writeStream.format("console").outputMode(OutputMode.Complete())

  query.start().awaitTermination()
}
