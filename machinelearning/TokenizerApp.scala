package project1.machinelearning

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, RegexTokenizer, StopWordsRemover, Tokenizer}
import org.apache.spark.sql.SparkSession

object TokenizerApp extends App{

  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder().appName("TokenizerApp")
    .master("local[*]")
    .getOrCreate()

  val sentence = spark.createDataFrame(Seq(
    (0, "Tokenization, is the process of enchanting words, from the raw text"),
    (1, "If you want ot have more advances Tokenization, RegexTokenizer, is a good option"),
    (2, "Here we will provide a sample to tokenize a sentences"),
    (3, "This way, you can find all matching occurences")
  )).toDF("id", "sentence")

  // creaate a tokenizer by instantiating the Tokenizer
  val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
  import org.apache.spark.sql.functions._

  // count the number of tokens in each sentence using UDF
  val countTokens = udf { (words: Seq[String]) => words.length }

  // tokenize words from each sentence
  val tokenized = tokenizer.transform(sentence)
  tokenized.show(false)

  // show each token against raw sentences
  tokenized.select("sentence", "words")
    .withColumn("tokens", countTokens(col("words")))
    .show(false)

  // get features from tokenized data
  val cvModel3: CountVectorizerModel = new CountVectorizer()
    .setInputCol("words")
    .setOutputCol("features")
    .setVocabSize(3)
    .setMinDF(2)
    .fit(tokenized)

  val features3 = cvModel3.transform(tokenized)
//  features3.show(false)

  // using Regex Tokenizer

  val regexTokenizer = new RegexTokenizer()
    .setInputCol("sentence")
    .setOutputCol("words")
    .setPattern("\\W+")
    .setGaps(true)

  val regexTokenized = regexTokenizer.transform(sentence)
  regexTokenized.select("sentence", "words")
    .withColumn("tokens", countTokens(col("words")))
    .show(false)

  // stop Word remover
  // stop words are words that should be excluded from the input,
  // typically because the words appear frequently and don't carry as much meaning

  val remover = new StopWordsRemover()
    .setInputCol("words")
    .setOutputCol("filtered")

  val newDF = remover.transform(regexTokenized)
  newDF.select("id", "filtered").show(false)


}
