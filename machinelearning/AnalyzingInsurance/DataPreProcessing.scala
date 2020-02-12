package project1.machinelearning.AnalyzingInsurance


import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import org.apache.spark.ml.feature.VectorAssembler

object DataPreProcessing {
  var trainSample = 1.0
  var testSample = 1.0
  val data = trainInput.withColumnRenamed("loss", "label")
    .sample(false,trainSample).na.drop
  val seed = 12345L
  val splits = data.randomSplit(Array(0.75,0.25), seed)
  val (trainingData, validationData) = (splits(0), splits(1))
  trainingData.cache
  validationData.cache

  val testData = testInput.sample(false,testSample).na.drop.cache
  // Since the training set contains both the numerical and categorical values, we need to
  //identify and treat them separately

  import Utils._
  val featureCols = trainingData.columns
    .filter(removeTooManyCateg)
    .filter(onlyFeatureCol)
    .map(categNewCols)

  // StringIndexer encodes a given string column of labels to a column of label indices
  // If the input column is numeric in nature, we cast it to string using the StringIndexer
  // and index the string values When downstream pipeline components such as Estimator or
  // Transformer make use of this string-indexed label,
  // you must set the input column of the component to this string-indexed column name.
  // we need to use the StringIndexer() for categorical columns

  val stringIndexerStages = trainingData.columns.filter(isCateg)
    .map(c => new StringIndexer()
      .setInputCol(c)
      .setOutputCol(categNewCols(c))
      .fit(trainInput.select(c).union(testInput.select(c)))
    )
  /*
   Note that this is not an efficient approach. An alternative approach would be
   using a OneHotEncoder estimator. OneHotEncoder maps a column of label indices to
   a column of binary vectors, with a single one-value at most.
   This encoding permits algorithms that expect
   continuous features, such as logistic regression, to utilize categorical features.
    */
  // Now let's use the VectorAssembler() to transform a given list of columns into a single vector column:

  val assembler = new VectorAssembler()
    .setInputCols(featureCols)
    .setOutputCol("features")
  /*
  VectorAssembler is a transformer. It combines a given list of columns into a single vector column.
  It is useful for combining the raw features and   features generated by different feature transformers
  into one feature vector, in order to train ML models such as logistic regression and decision trees.
*/
}