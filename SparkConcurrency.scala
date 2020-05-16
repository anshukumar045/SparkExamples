package scala.com.kanshu.main
import scala.util._ 
import scala.concurrent._
import scala.concurrent.duration.Duration
import scala.concurrent.duration.Duration._
import scala.concurrent.ExecutionContext.Implicits.global

import org.apache.log4j.{Level,Logger}
import org.apache.spark.sql.SparkSession

object SparkConcurrencyApp extends App{
  
  Logger.getLogger("org").setLevel(Level.ERROR)
  
  def slowFoo[T](x: T): T = {
    println(s"slowFoo start ($x)")
    Thread.sleep(5000)
    println(s"slowFoo ends($x)")
    x
  }
  
  def fastfoo[T](x: T): T = {
    println(s"fastFoo($x)")
    x
  }
  
  val sc = SparkSession.builder().appName("SparkConcurrency").master("local[*]").getOrCreate()
  
//  sc.sparkContext.parallelize(1 to 3).map(fastfoo).map(slowFoo).collect
  
  // Concurrency with Future
  /* Futures are a means of doing asynchronous programming in Scala.
   They provide a native way for us to express concurrent actions without having to
   deal with the nitty gritty of actually setting up threads.
   Creating an object to accomplish this allows us to switch the executionContext for this code 
   and allow us to control the concurrency at a JVM level. Each executor JVM will only make one instance 
   of this ConcurrentContext which means if we switch to a thread pool 
   based ExecutionContext it can be shared by all tasks running on the same machine.
   This context will use the global execution context on the executor to do work on many threads at a time.
   Our executeAsync method simply takes a code block and transforms it into a task to be run on the
   global context (implicitly imported and set). The result is left in a Future object which 
   we can poll at our leisure.If we wrap our slowFoo in the executeAsync we see that all the tasks are 
   queued and our code returns immediately.*/
  object ConcurrentContext {
    import scala.util._ 
    import scala.concurrent._
    import scala.concurrent.duration.Duration
    import scala.concurrent.duration.Duration._
    import scala.concurrent.ExecutionContext.Implicits.global
    
    /* Wraps a code block in a Future and returns a Future
      Await an entire sequence of futures and return an iterator. This will wait for all futures to 
      a complete before returning
      While it’s nice that everything is going on in parallel this has a few fatal flaws.
      We have no form of error handling. If something goes wrong we are basically lost and 
      Spark will not retry the task. The Spark task completes before all the work is done. 
      This means that if the application shut down it could close out our executors while they are 
      still processing work. We have no way of feeding the results of this asynchronous work into 
      another process meaning we lose the results as well. Unlimited futures flying around can lead to 
      the overhead of their managment decreasing throughput.
      To fix 1 - 3 we need to actually end up waiting on our Futures to complete but this is where 
      things start getting tricky. */
    
    def executeAsync[T](f: => T): Future[T] = {
     Future(f) 
    }
    
    /* Method awaitAll to actually make sure that our futures finish and 
     give us back the resultant values. This means that our Spark Job will not 
     complete until all of our Futures have completed as we see in the following run 
     The code behind Futures.sequence will greedly grab all of our futures at once. 
    This means the amount of concurrent work that is being run is unbounded. 
    This issue is compounded by the fact that we need to hold the results of all 
    the futures in memory at the same time before we return any of the results. 
    In practical terms this means that you will have OOM’s if the set we are waiting 
    on here is large. For example, rdd.map(parallelThing).filter(x < 1).count would 
    benefit from filtering the records as they are calculated rather than getting all the 
    values then filtering.  */
    
    def awaitAll[T](it: Iterator[Future[T]], timeout: Duration = Inf) = {
      Await.result(Future.sequence(it), timeout)
    }
    
    /* Awaits only a set of elements at a time. At Most batchSize futures will ever  be in memory at a time
      Now instead of waiting for the entire task at once we break it up into chunks then await each chunk. 
      This means we will never have more than batchSize tasks in progress at a time solving our 
      “unbounded parallelism” and “having all records in memory” issues.
      Notice how we are only ever working on 3 elements at a time? 
      Unfortunately we are blocking on every batch. 
      Every batch must wait for the previous batch to completely finish before any of the 
      work in the next batch can start. We can do better than this with a few different methods. 
      Here is one possibility where the ordering is maintained but we keep a rolling buffer 
      of futures to be completed.
     */
    def awaitBatch[T](it: Iterator[Future[T]], batchSize: Int = 3, timeout: Duration = Inf) = {
      it.grouped(batchSize)
      .map(batch => Future.sequence(batch))
      .flatMap(futureBatch => Await.result(futureBatch, timeout))
    }
    
    /* Awaits only a set of elements at a time. Instead of waiting for the entire batch
      to finish waits only for the head element before requesting the next future
      This gives us a order maintaining buffered set of Futures. The sliding batch makes sure 
      that the iterator has queued up to batchSize futures while still making sure that we pass 
      along the head element as soon as we can. The span is required to make sure that we wait on 
      the last sliding window for it to finish completely.
     */
    
    def awaitSliding[T](it: Iterator[Future[T]], batchSize: Int = 3, timeout: Duration = Inf): Iterator[T] = {
      val slidingIterator = it.sliding(batchSize - 1).withPartial(true) //Our look ahead (hasNext) will auto start the nth future in the batch
    
      val (initIterator, tailIterator) = slidingIterator.span(_ => slidingIterator.hasNext)
        initIterator.map( futureBatch => Await.result(futureBatch.head, timeout)) ++
        tailIterator.flatMap( lastBatch => Await.result(Future.sequence(lastBatch), timeout))
    }
  }
  
  println("Concurrecny with Future")
  /*
  sc.sparkContext.parallelize(1 to 3)
  .map(fastfoo).map(x => ConcurrentContext.executeAsync(slowFoo(x)))
  .mapPartitions(it => ConcurrentContext.awaitAll(it))
  .foreach(x => println(s"Finishing wiht $x"))
  
  */
  
  sc.sparkContext.parallelize(1 to 5)
  .map(fastfoo).map(x => ConcurrentContext.executeAsync(slowFoo(x)))
  .mapPartitions(it => ConcurrentContext.awaitBatch(it))
  .foreach(x => println(s"Finishing wiht $x"))
  
  sc.sparkContext.parallelize(1 to 5).map(fastfoo)
  .map(x => ConcurrentContext.executeAsync(slowFoo(x)))
  .mapPartitions(it => ConcurrentContext.awaitSliding(it))
  .foreach(x => println(s"Finishing with $x"))

}
