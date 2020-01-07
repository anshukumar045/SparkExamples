package project1.machinelearning
//Params for [[CountVectorizer]] and [[CountVectorizerModel]].
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}

object CountVectorizerParamsApp extends App{
  Logger.getLogger("org").setLevel(Level.ERROR)
}
private[machinelearning]trait CountVectorizerParams extends Params with HasInputCol with HasOutputCol {

  //CountVectorizer will build a vocabulary that only considers
  // the top vocabSize terms ordered by term frequency across the corpus
  val vocabSize: IntParam =
  new IntParam(this, "vocabSize", "max size of vocabulary", ParamValidators.gt(0))

  def getVocabSize: Int = $(vocabSize)

  // Specifies the minimum number of different documents a term must appear in to be included in the vocabulary
  // If this is an integer greater than or equal to 1, this specifies the number of documents
  // the term must appear in; if this is a double in [0,1), then this specifies the fraction of documents

  val minDF: DoubleParam = new DoubleParam(this, "minDF", "Specifies the min number of " +
    " different documents a term must appear in to be included in the vocabulary." +
    " If this is an integer >= 1, this specifies the number of documents the term must" +
    " appear in; if this is a double in [0,1), then this specifies the fraction of documents.",
    ParamValidators.gtEq(0.0))

  def getMinDF: Double = $(minDF)

  // Specifies the maximum number of different documents a term could appear in to be included in the vocabulary
  // A term that appears more than the threshold will be ignored. If this is an  integer greater than or equal to 1,
  // this specifies the maximum number of documents the term could appear in; if this is a double in [0,1),
  // then this specifies the maximum fraction of documents the term could appear in.

  val maxDF: DoubleParam = new DoubleParam(this, "maxDF", "Specifies the maximum number of" +
    " different documents a term could appear in to be included in the vocabulary." +
    " A term that appears more than the threshold will be ignored. If this is an integer >= 1," +
    " this specifies the maximum number of documents the term could appear in;" +
    " if this is a double in [0,1), then this specifies the maximum fraction of" +
    " documents the term could appear in.",
    ParamValidators.gtEq(0.0))

  def getMaxDF: Double = $(maxDF)

}

