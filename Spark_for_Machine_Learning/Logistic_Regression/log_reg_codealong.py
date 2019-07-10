# Databricks notebook source
df = spark.sql("SELECT * FROM titanic_csv")

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.columns

# COMMAND ----------

cols = df.select(["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])

# COMMAND ----------

data = cols.na.drop()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
import math
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, VectorIndexer, OneHotEncoder, StringIndexer

# COMMAND ----------

gender_indexer = StringIndexer(inputCol="Sex",outputCol="SexIndex")

# COMMAND ----------

gender_encoder = OneHotEncoder(inputCol="SexIndex", outputCol="SexVec")

# COMMAND ----------

embark_indexer = StringIndexer(inputCol="Embarked", outputCol="EmbarkIndex")

# COMMAND ----------

embark_encoder = OneHotEncoder(inputCol="EmbarkIndex",outputCol="EmbarkVec")

# COMMAND ----------

assembler = VectorAssembler(inputCols=["Pclass","SexVec","EmbarkVec","Age","SibSp","Parch","Fare"], outputCol = 'features')

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# COMMAND ----------

from pyspark.ml import Pipeline

# COMMAND ----------

log_reg_titanic = LogisticRegression(labelCol='Survived')

# COMMAND ----------

pipeline = Pipeline(stages=[gender_indexer,embark_indexer,gender_encoder,embark_encoder,assembler,log_reg_titanic])

# COMMAND ----------

train,test = data.randomSplit([.7,.3])

# COMMAND ----------

model = pipeline.fit(train)

# COMMAND ----------

results = model.transform(test)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol="Survived")

# COMMAND ----------

eval.evaluate(results)

# COMMAND ----------

results.select("Survived","prediction").show()

# COMMAND ----------

under_roc =eval.evaluate(results)

# COMMAND ----------

under_roc

# COMMAND ----------


