from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Basics').getOrCreate()

df = spark.read.json('Spark_DataFrames/people.json')
df.show()
df.printSchema()
print(df.columns)
df.describe().show()
