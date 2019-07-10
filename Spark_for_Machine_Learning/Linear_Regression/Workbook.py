from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
spark =SparkSession.builder.appName('lrex').getOrCreate()

# #Learning
# training = spark.read.format('libsvm').load('Spark_for_Machine_Learning/Linear_Regression/sample_linear_regression_data.txt')
# training.show()
# '''
# +-------------------+--------------------+
# |              label|            features|
# +-------------------+--------------------+
# | -9.490009878824548|(10,[0,1,2,3,4,5,...|
# | 0.2577820163584905|(10,[0,1,2,3,4,5,...|
# | -4.438869807456516|(10,[0,1,2,3,4,5,...|
# |-19.782762789614537|(10,[0,1,2,3,4,5,...|
# | -7.966593841555266|(10,[0,1,2,3,4,5,...|
# | -7.896274316726144|(10,[0,1,2,3,4,5,...|
# | -8.464803554195287|(10,[0,1,2,3,4,5,...|
# | 2.1214592666251364|(10,[0,1,2,3,4,5,...|
# | 1.0720117616524107|(10,[0,1,2,3,4,5,...|
# |-13.772441561702871|(10,[0,1,2,3,4,5,...|
# | -5.082010756207233|(10,[0,1,2,3,4,5,...|
# |  7.887786536531237|(10,[0,1,2,3,4,5,...|
# | 14.323146365332388|(10,[0,1,2,3,4,5,...|
# |-20.057482615789212|(10,[0,1,2,3,4,5,...|
# |-0.8995693247765151|(10,[0,1,2,3,4,5,...|
# | -19.16829262296376|(10,[0,1,2,3,4,5,...|
# |  5.601801561245534|(10,[0,1,2,3,4,5,...|
# |-3.2256352187273354|(10,[0,1,2,3,4,5,...|
# | 1.5299675726687754|(10,[0,1,2,3,4,5,...|
# | -0.250102447941961|(10,[0,1,2,3,4,5,...|
# +-------------------+--------------------+
# '''
# lr = LinearRegression()
# lrModel = lr.fit(training)
# print(lrModel.coefficients)
# training_summary = lrModel.summary
# print(training_summary.r2)
# print(training_summary.rootMeanSquaredError)
# df1,df2 = training.randomSplit([.5,.5])
# trainModel = lr.fit(df1)
# df1.describe().show()
# df2.describe().show()
# results = trainModel.evaluate(df2)
#
# unlabeled_data = df2.select("features")
# df3 = trainModel.transform(unlabeled_data)
# df3.show()

#First Example
data = spark.read.csv('Spark_for_Machine_Learning/Linear_Regression/Ecommerce_Customers.csv',inferSchema=True, header=True)
assembler = VectorAssembler(inputCols=['Avg Session Length','Time on Website','Length of Membership', 'Time on App'],outputCol='features')
output = assembler.transform(data)
print(output.head(1))
final_data = output.select('features', 'Yearly Amount Spent')
final_data.show()
training,test = final_data.randomSplit([.7,.3])
lr = LinearRegression(labelCol='Yearly Amount Spent')
model = lr.fit(training)
summary = model.evaluate(test)
print(summary.r2)
print(summary.rootMeanSquaredError)
print('Prediction:')
model.transform(test.select('features')).show()
print('Actual:')
test.show()

spark.stop()
