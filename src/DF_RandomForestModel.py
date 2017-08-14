# Databricks notebook source
import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.mllib.util import MLUtils
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

tpath = '/FileStore/tables/1fog75651502067187082/libsvm_train.csv'
tdata = spark.read.format("libsvm").load(tpath)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
tfeatureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(tdata)

# Split the data into training and test sets (30% held out for testing)
(ttrainingData, ttestData) = tdata.randomSplit([0.7, 0.3])

# Train a RandomForest model.
trf = RandomForestRegressor(featuresCol="indexedFeatures")

# Chain indexer and forest in a Pipeline
tpipeline = Pipeline(stages=[tfeatureIndexer, trf])

# Train model.  This also runs the indexer.
tmodel = tpipeline.fit(ttrainingData)

# Make predictions.
tpredictions = tmodel.transform(ttestData)

# Select example rows to display.
tpredictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

rfModel = model.stages[1]
print(rfModel)  # summary only

# COMMAND ----------

tpredictions.select("prediction", "label", "features").show(500)

# COMMAND ----------

pre_xan = '/FileStore/tables/jeexcosw1502067610877/xab'
pdata_xan = spark.read.format("libsvm").load(pre_xan)
predict_xan = tmodel.transform(pdata_xan)
predict_xan.select("prediction", "label", "features").show(5000)
