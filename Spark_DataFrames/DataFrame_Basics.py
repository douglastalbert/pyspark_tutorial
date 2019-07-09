from pyspark.sql import SparkSession
from pyspark.sql.types import (StructField,StringType,
                                IntegerType,StructType)
from pyspark.sql.functions import (stddev, format_number,
                                    mean, dayofmonth, hour,
                                    dayofyear, month, year,
                                    weekofyear, date_format)

spark = SparkSession.builder.appName('Basics').getOrCreate()

#DataFrame Basics

data_schema = [StructField('age',IntegerType(),True),
                StructField('name',StringType(),True)]
final_struc = StructType(fields=data_schema)

df = spark.read.json('Spark_DataFrames/people.json',schema=final_struc)
df.show()
df.printSchema()

print(df.columns)
df.describe().show()

#Part Two

df.select('age').show()
print(df.head(2))

#Basic Operations

df1 = spark.read.csv('Spark_DataFrames/appl_stock.csv', inferSchema=True, header=True)
df1.show()
df1.describe().show()

df1.filter(df1['Adj_Close']>100).show()
df1.select(["Date","Volume"]).show()
qResult = df1.filter(df1['Low']==197.16).collect()
print(qResult)

#GroupBy and Aggregate Operations

df2 = spark.read.csv('Spark_DataFrames/sales_info.csv',inferSchema=True,header=True)
df2.groupBy('Company').mean().show()
df2.agg({'Sales':'mean'}).show()

sales_std = df2.select(stddev("Sales").alias('std'))
sales_std.show()
sales_std.select(format_number('std',2)).show()

df2.orderBy("Sales").show()
df2.orderBy(df2["Sales"].desc()).show()

#Missing Data
df3 = spark.read.csv('Spark_DataFrames/ContainsNull.csv',header=True,inferSchema=True)

df3.show()

meanRow = df3.select(mean(df3['Sales'])).collect()
meanSales = meanRow[0][0]
df3.na.fill(meanSales,['Sales']).show()
df3.na.drop(thresh=2).show()

#Dates and Timestamps

df4 = spark.read.csv("Spark_DataFrames/appl_stock.csv",header=True,inferSchema=True)
df4.select(['Date','Open']).show()
df4.select(dayofmonth(df4['Date'])).show()
df4Y = df4.withColumn("Year", year(df4["Date"]))
df4Y.groupBy("Year").mean().select(["Year", "avg(Close)"]).show()
