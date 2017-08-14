# Databricks notebook source
import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.mllib import *


filePath = '/FileStore/tables/j7j62ggi1502033614666/DataSet.csv'

# COMMAND ----------

labelDataSchema = StructType([StructField("id_parcel", DoubleType(), True), 
                              StructField("aircon", StringType(), True),
                              StructField("architectural_style", StringType(), True), 
                              StructField("area_basement", DoubleType(), True),
                              StructField("num_bathroom", DoubleType(), True), 
                              StructField("num_bedroom", DoubleType(), True),
                              StructField("framing", StringType(), True), 
                              StructField("quality", StringType(), True),
                              StructField("num_bathroom_calc", DoubleType(), True), 
                              StructField("deck", StringType(), True),
                              StructField("area_firstfloor_finished", DoubleType(), True), 
                              StructField("area_total_calc", DoubleType(), True),
                              StructField("area_live_finished", DoubleType(), True), 
                              StructField("area_liveperi_finished", DoubleType(), True),
                              StructField("area_total_finished", DoubleType(), True), 
                              StructField("area_unknown", DoubleType(), True),
                              StructField("area_base", DoubleType(), True), 
                              StructField("no_fip", StringType(), True),
                              StructField("num_fireplace", DoubleType(), True), 
                              StructField("num_bath", DoubleType(), True),
                              StructField("num_garage", DoubleType(), True), 
                              StructField("area_garage", DoubleType(), True),
                              StructField("flag_tub", StringType(), True), 
                              StructField("heating", StringType(), True),
                              StructField("latitude", DoubleType(), True), 
                              StructField("longitude", DoubleType(), True),
                              StructField("area_lot", DoubleType(), True), 
                              StructField("num_pool", DoubleType(), True),
                              StructField("area_pool", DoubleType(), True), 
                              StructField("s_h", StringType(), True),
                              StructField("pool_sh", StringType(), True), 
                              StructField("pool_nosh", StringType(), True),
                              StructField("zoning_landuse_county", StringType(), True), 
                              StructField("zoning_landuse", StringType(), True),
                              StructField("zoning_property", StringType(), True), 
                              StructField("rawcensus", DoubleType(), True),
                              StructField("region_city", StringType(), True), 
                              StructField("region_county", StringType(), True),
                              StructField("region_neighbor", StringType(), True), 
                              StructField("region_zip", StringType(), True),
                              StructField("num_room", DoubleType(), True), 
                              StructField("story", DoubleType(), True),
                              StructField("num_75_bath", DoubleType(), True), 
                              StructField("material", StringType(), True),
                              StructField("num_unit", DoubleType(), True), 
                              StructField("area_patio", DoubleType(), True),
                              StructField("area_shed", DoubleType(), True), 
                              StructField("build_year", StringType(), True),
                              StructField("num_story", DoubleType(), True), 
                              StructField("flag_fireplace", StringType(), True),
                              StructField("tax_building", DoubleType(), True), 
                              StructField("tax_total", DoubleType(), True),
                              StructField("tax_year", DoubleType(), True), 
                              StructField("tax_land", DoubleType(), True),
                              StructField("tax_property", DoubleType(), True), 
                              StructField("tax_delinquency", DoubleType(), True),
                              StructField("tax_delinquency_year", DoubleType(), True), 
                              StructField("census", DoubleType(), True),
                              StructField("transac", StringType(), True), 
                              StructField("logerror", DoubleType(), True),
                             ])

dataSet = spark.read.format("com.databricks.spark.csv").option("header", "true").schema(labelDataSchema).load(filePath)


# COMMAND ----------

dataSet.show()

# COMMAND ----------

newDataSet = dataSet.drop("area_basement").drop("area_patio").drop("area_shed").drop("area_pool").drop("area_firstfloor_finished").drop("area_base").drop("area_live_finished").drop("area_liveperi_finished").drop("area_total_finished").drop("area_unknown").drop("num_bathroom_calc").drop("num_bath").drop("num_75_bath").drop("num_fireplace").drop("num_pool").drop("region_county").drop("tax_total").drop("tax_delinquency_year").drop("framing").drop("material").drop("deck").drop("story").drop("architectural_style").drop("s_h").drop("pool_sh").drop("pool_nosh").drop("rawcensus").drop("census").drop("flag_fireplace").drop("tax_delinquency").drop("flag_tub")

# COMMAND ----------

newDataSet.printSchema()

# COMMAND ----------

title_list = map(str.strip, ['aircon',  'quality',  'heating',  'zoning_landuse_county', 'zoning_landuse', 'zoning_property', 'region_city', 'region_neighbor', 'region_zip', 'transac'])
title_list

# COMMAND ----------

none_fill = dict(zip(title_list, ['unknown']*len(title_list)))
newDataSet = newDataSet.na.fill(none_fill)

# COMMAND ----------

att = list()
for i,title in enumerate(title_list):
  att = att + newDataSet.select(title).distinct().rdd.flatMap(lambda x: x).collect()

# COMMAND ----------

atts = list(set(att))
n_atts = map(str,range(len(atts)))
atts

# COMMAND ----------

for title in title_list:
  newDataSet = newDataSet.na.replace(atts, n_atts, title)

# COMMAND ----------

newDataSet.show()

# COMMAND ----------

num_title = ['num_bathroom', 'num_bedroom', 'area_total_calc', 'num_garage', 'area_garage', 'latitude', 'longitude', 'area_lot', 'num_room', 'num_unit', 'num_story', 'tax_building', 'tax_year', 'tax_land', 'tax_property']

num_fill = dict()

for title in num_title:
  average = newDataSet.groupBy().avg(title).collect()[0][0]
  num_fill[title] = average

num_fill

# COMMAND ----------

newDataSet = newDataSet.na.fill(num_fill)

# COMMAND ----------

newDataSet.show()

# COMMAND ----------

newDataSet.printSchema()

# COMMAND ----------

changeList = ['aircon',  'quality',  'heating',  'zoning_landuse_county', 'zoning_landuse', 'zoning_property', 'region_city', 'region_neighbor', 'region_zip', 'build_year', 'transac', 'no_fip']

for title in changeList:
  newDataSet = newDataSet.withColumn(title, newDataSet[title].cast(DoubleType()))

# COMMAND ----------

newDataSet.printSchema()

# COMMAND ----------

DataSetRDD = newDataSet.rdd.map(list)
DataSetRDD.collect()

# COMMAND ----------

x = DataSetRDD.take(1)
x

# COMMAND ----------

x[0][:-1]

# COMMAND ----------

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors

labeledPoints = DataSetRDD.map(lambda x : LabeledPoint(x[-1], Vectors.dense(x[:-1])))

(trainingData, testData) = labeledPoints.randomSplit([0.7, 0.3])

# COMMAND ----------

labeledPoints.collect()

# COMMAND ----------

from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils
# Train a GradientBoostedTrees model.
#  Notes: (a) Empty categoricalFeaturesInfo indicates all features are continuous.
#         (b) Use more iterations in practice.
model = GradientBoostedTrees.trainRegressor(trainingData,
                                            categoricalFeaturesInfo={}, numIterations=3)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())
print('Test Mean Squared Error = ' + str(testMSE))
print('Learned regression GBT model:')
print(model.toDebugString())

# COMMAND ----------

from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils
# Train a GradientBoostedTrees model.
#  Notes: (a) Empty categoricalFeaturesInfo indicates all features are continuous.
#         (b) Use more iterations in practice.
model1 = GradientBoostedTrees.trainRegressor(trainingData,
                                            categoricalFeaturesInfo={}, numIterations=5)

# Evaluate model on test instances and compute test error
predictions1 = model1.predict(testData.map(lambda x: x.features))
labelsAndPredictions1 = testData.map(lambda lp: lp.label).zip(predictions1)
testMSE1 = labelsAndPredictions1.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())
print('Test Mean Squared Error = ' + str(testMSE1))
print('Learned regression GBT model1:')
print(model1.toDebugString())

# COMMAND ----------

from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils
# Train a GradientBoostedTrees model.
#  Notes: (a) Empty categoricalFeaturesInfo indicates all features are continuous.
#         (b) Use more iterations in practice.
model2 = GradientBoostedTrees.trainRegressor(trainingData,
                                            categoricalFeaturesInfo={}, numIterations=7)

# Evaluate model on test instances and compute test error
predictions2 = model2.predict(testData.map(lambda x: x.features))
labelsAndPredictions2 = testData.map(lambda lp: lp.label).zip(predictions2)
testMSE2 = labelsAndPredictions2.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())
print('Test Mean Squared Error = ' + str(testMSE2))
print('Learned regression GBT model2:')
print(model2.toDebugString())

# COMMAND ----------


