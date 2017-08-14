# Databricks notebook source
import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.mllib import *
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors

# COMMAND ----------

property_file = "/FileStore/tables/2wyd446l1502045861418/properties_2016.csv"
training_file = "/FileStore/tables/bdlfhd021502051248601/n_train_2016_v2.csv"

# COMMAND ----------

labelDataSchema = StructType([StructField("id_parcel", StringType(), True), 
                              StructField("aircon", StringType(), True),
                              StructField("architectural_style", StringType(), True), 
                              StructField("area_basement", StringType(), True),
                              StructField("num_bathroom", StringType(), True), 
                              StructField("num_bedroom", StringType(), True),
                              StructField("framing", StringType(), True), 
                              StructField("quality", StringType(), True),
                              StructField("num_bathroom_calc", StringType(), True), 
                              StructField("deck", StringType(), True),
                              StructField("area_firstfloor_finished", StringType(), True), 
                              StructField("area_total_calc", StringType(), True),
                              StructField("area_live_finished", StringType(), True), 
                              StructField("area_liveperi_finished", StringType(), True),
                              StructField("area_total_finished", StringType(), True), 
                              StructField("area_unknown", StringType(), True),
                              StructField("area_base", StringType(), True), 
                              StructField("no_fip", StringType(), True),
                              StructField("num_fireplace", StringType(), True), 
                              StructField("num_bath", StringType(), True),
                              StructField("num_garage", StringType(), True), 
                              StructField("area_garage", StringType(), True),
                              StructField("flag_tub", StringType(), True), 
                              StructField("heating", StringType(), True),
                              StructField("latitude", StringType(), True), 
                              StructField("longitude", StringType(), True),
                              StructField("area_lot", StringType(), True), 
                              StructField("num_pool", StringType(), True),
                              StructField("area_pool", StringType(), True), 
                              StructField("s_h", StringType(), True),
                              StructField("pool_sh", StringType(), True), 
                              StructField("pool_nosh", StringType(), True),
                              StructField("zoning_landuse_county", StringType(), True), 
                              StructField("zoning_landuse", StringType(), True),
                              StructField("zoning_property", StringType(), True), 
                              StructField("rawcensus", StringType(), True),
                              StructField("region_city", StringType(), True), 
                              StructField("region_county", StringType(), True),
                              StructField("region_neighbor", StringType(), True), 
                              StructField("region_zip", StringType(), True),
                              StructField("num_room", StringType(), True), 
                              StructField("story", StringType(), True),
                              StructField("num_75_bath", StringType(), True), 
                              StructField("material", StringType(), True),
                              StructField("num_unit", StringType(), True), 
                              StructField("area_patio", StringType(), True),
                              StructField("area_shed", StringType(), True), 
                              StructField("build_year", StringType(), True),
                              StructField("num_story", StringType(), True), 
                              StructField("flag_fireplace", StringType(), True),
                              StructField("tax_building", StringType(), True), 
                              StructField("tax_total", StringType(), True),
                              StructField("tax_year", StringType(), True), 
                              StructField("tax_land", StringType(), True),
                              StructField("tax_property", StringType(), True), 
                              StructField("tax_delinquency", StringType(), True),
                              StructField("tax_delinquency_year", StringType(), True), 
                              StructField("census", StringType(), True)
                             ])
dataSet1 = spark.read.format("com.databricks.spark.csv").option("header", "true").schema(labelDataSchema).load(property_file)

# COMMAND ----------

dataSet1.printSchema()

# COMMAND ----------

dataSet1.show()

# COMMAND ----------

dataSet1.count()

# COMMAND ----------

newDataSet = dataSet1.drop("area_basement").drop("area_patio").drop("area_shed").drop("area_pool").drop("area_firstfloor_finished").drop("area_base").drop("area_live_finished").drop("area_liveperi_finished").drop("area_total_finished").drop("area_unknown").drop("num_bathroom_calc").drop("num_bath").drop("num_75_bath").drop("num_fireplace").drop("num_pool").drop("region_county").drop("tax_total").drop("tax_delinquency_year").drop("framing").drop("material").drop("deck").drop("story").drop("architectural_style").drop("s_h").drop("pool_sh").drop("pool_nosh").drop("rawcensus").drop("census").drop("flag_fireplace").drop("tax_delinquency").drop("flag_tub")


# COMMAND ----------

newDataSet.show()

# COMMAND ----------

transac_schema = StructType([StructField("idparcel", StringType(), True), 
                              StructField("logerror", StringType(), True),
                              StructField("transac", StringType(), True),])
transac_data = spark.read.format("com.databricks.spark.csv").option("header", "true").schema(transac_schema).load(training_file)
transac_data = transac_data.withColumn('idparcel', transac_data['idparcel'].cast(DoubleType()))
transac_data = transac_data.withColumn('logerror', transac_data['logerror'].cast(DoubleType()))
transac_data = transac_data.withColumn('transac', transac_data['transac'].cast(DoubleType()))

# COMMAND ----------

transac_data.show()
transac_data.count()

# COMMAND ----------

data = transac_data.join(newDataSet, newDataSet.id_parcel == transac_data.idparcel, 'left').drop('idparcel')

# COMMAND ----------

## impute missing values and transfer categorical features into numeric features
string_title = map(str.strip, ['aircon',  'quality',  'heating',  'zoning_landuse_county', 'zoning_landuse', 'zoning_property', 'region_city', 'region_neighbor', 'region_zip', 'build_year', 'no_fip'])

none_fill = dict(zip(string_title, ['unknown']*len(string_title)))

newDataSet = data.na.fill(none_fill)

att = list()
for i,title in enumerate(string_title):
  att = att + newDataSet.select(title).distinct().rdd.flatMap(lambda x: x).collect()

atts = list(set(att))
n_atts = map(str,range(len(atts)))

for title in string_title:
  newDataSet = newDataSet.na.replace(atts, n_atts, title)
  
num_title = ['id_parcel', 'num_bathroom', 'num_bedroom', 'area_total_calc', 'num_garage', 'area_garage', 'latitude', 'longitude', 'area_lot', 'num_room', 'num_unit', 'num_story', 'tax_building', 'tax_year', 'tax_land', 'tax_property']

for title in num_title:
  newDataSet = newDataSet.withColumn(title, newDataSet[title].cast(DoubleType()))
  
num_fill = dict()

for title in num_title:
  average = newDataSet.groupBy().avg(title).collect()[0][0]
  num_fill[title] = average
  
newDataSet = newDataSet.na.fill(num_fill)

for title in string_title:
  newDataSet = newDataSet.withColumn(title, newDataSet[title].cast(DoubleType()))

# COMMAND ----------

newDataSet.printSchema()

# COMMAND ----------

newDataSet.take(1)

# COMMAND ----------

data_rdd = newDataSet.rdd.map(list)

# COMMAND ----------

data_rdd.take(1)

# COMMAND ----------

## product a labeledPoints and split data into training data and test data
labeledPoints = data_rdd.map(lambda x : LabeledPoint(x[0], Vectors.dense(x[1:])))
labeledPoints.collect()

# COMMAND ----------

(trainingData, testData) = labeledPoints.randomSplit([0.7, 0.3])

# COMMAND ----------

## use random forest regression to train data, build a model and predict 
model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo={}, numTrees=7, featureSubsetStrategy="auto", impurity='variance', maxDepth=8, maxBins=32)
predictions = model.predict(testData.map(lambda x: x.features))

# COMMAND ----------

## estimate use MSE
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testMSE = labelsAndPredictions.map(lambda lp: (lp[0] - lp[1]) * (lp[0] - lp[1])).sum() / float(testData.count())
print('Test Mean Squared Error = ' + str(testMSE))
print('Learned regression forest model:')
print(model.toDebugString())

# COMMAND ----------

# Save model
model.save(sc, "target/tmp/RandomForestRegressionModel")

# COMMAND ----------

## use newDataSet(all data) to do predictions for Januray 2016-01
newDataSet1 = newDataSet.withColumn("transac", lit(1))
## impute missing values and transfer categorical features into numeric features
string_title = map(str.strip, ['aircon',  'quality',  'heating',  'zoning_landuse_county', 'zoning_landuse', 'zoning_property', 'region_city', 'region_neighbor', 'region_zip', 'build_year', 'no_fip'])

none_fill = dict(zip(string_title, ['unknown']*len(string_title)))

newDataSet1 = newDataSet1.na.fill(none_fill)

att = list()
for i,title in enumerate(string_title):
  att = att + newDataSet1.select(title).distinct().rdd.flatMap(lambda x: x).collect()

atts = list(set(att))
n_atts = map(str,range(len(atts)))

for title in string_title:
  newDataSet1 = newDataSet1.na.replace(atts, n_atts, title)
  
num_title = ['id_parcel', 'num_bathroom', 'num_bedroom', 'area_total_calc', 'num_garage', 'area_garage', 'latitude', 'longitude', 'area_lot', 'num_room', 'num_unit', 'num_story', 'tax_building', 'tax_year', 'tax_land', 'tax_property']

for title in num_title:
  newDataSet1 = newDataSet1.withColumn(title, newDataSet1[title].cast(DoubleType()))
  
num_fill = dict()

for title in num_title:
  average = newDataSet1.groupBy().avg(title).collect()[0][0]
  num_fill[title] = average
  
newDataSet1 = newDataSet1.na.fill(num_fill)

for title in string_title:
  newDataSet1 = newDataSet1.withColumn(title, newDataSet1[title].cast(DoubleType()))  
newDataSet1.rdd.map(list)
new_labeledPoints = newDataSet_rdd.map(lambda x : LabeledPoint(0, Vectors.dense(x[0:])))
predictions = model.predict(new_labeledPoints.map(lambda x: x.features))

## product an output
id_predictions = newDataSet_rdd.map(lambda l: l[0]).zip(predictions)
