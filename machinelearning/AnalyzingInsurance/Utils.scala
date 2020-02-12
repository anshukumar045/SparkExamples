package project1.machinelearning.AnalyzingInsurance

object Utils {

  def isCateg(c: String): Boolean = c.startsWith("cat")
  def categNewCols(c: String): String = if (isCateg(c)) s"idx_${c}" else c

  //remove categorical columns with too many categories
  def removeTooManyCateg(c: String): Boolean = !(c matches
    "cat(109$|110$|112$|113$|116$)")
  def onlyFeatureCol(c: String): Boolean = !(c matches "id|label")

}
