from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
import math
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
spark = SparkSession.builder.appName('mylogreg').getOrCreate()

data = spark.read.format('libsvm').load('Spark_for_Machine_Learning/Logistic_Regression/sample_libsvm_data.txt')
data.show()

lgr = LogisticRegression()
# model = lgr.fit(data)
# model_summ = model.summary
# model_summ.predictions.show()

d1,d2 = data.randomSplit([.7,.3])
model = lgr.fit(d1)
eval = model.evaluate(d2)
evaluator = BinaryClassificationEvaluator()
under_roc = evaluator.evaluate(eval.predictions)
print(under_roc)
