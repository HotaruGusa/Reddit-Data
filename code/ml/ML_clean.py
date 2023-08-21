# Databricks notebook source
# MAGIC %md
# MAGIC ### Import Package

# COMMAND ----------

# upgrade pip
!pip install --upgrade pip

# COMMAND ----------

# import libraries
import os
import pandas as pd
import re
import pyspark.sql.functions as f
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.functions import col,sum,avg,max,count
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys
from collections import ChainMap
import plotly.graph_objects as go
from pyspark.sql.functions import array_remove
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
import sparknlp
from sparknlp.pretrained import PretrainedPipeline
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql.types import *
from pyspark.sql import Window
from pyspark.sql.functions import collect_list
from wordcloud import WordCloud, STOPWORDS
import re
from pyspark.sql.types import ArrayType,StringType,IntegerType

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read data and clean rows

# COMMAND ----------

# read from submission data
nlp_df = spark.read.parquet("/FileStore/ML_data")
nlp_df.printSchema()

# COMMAND ----------

# count the nlp
nlp_df.count()

# COMMAND ----------

# remove deleted and removed posts
nlp_df = nlp_df.filter(col('selftext') != "[deleted]").filter(col('selftext') != "[removed]")

# COMMAND ----------

# remove Na
nlp_df = nlp_df.filter(col('selftext') != '')\
.filter(~col('selftext').contains('None'))\
.filter(~col('selftext').contains('NULL'))\
.filter(~col('selftext').isNull())\
.filter(~isnan(col("selftext")))

# COMMAND ----------

# start pyspark
spark = sparknlp.start()
print("Spark NLP version", sparknlp.version())
print("Apache Spark version:", spark.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pipeline to clean data

# COMMAND ----------

# add extra stopwords
stopwords = list(STOPWORDS)
more_stop = ['mt', 'https', 'c', 'removed', 'deleted', '1', 'https:', '?','one', 'it', 'redd','2','&amp','x200b','HTTPS:','im','ive','dont','know','ampx200b','cant']
stopwords += more_stop

# COMMAND ----------

# NLP pipeline
document = DocumentAssembler()\
    .setInputCol("selftext")\
    .setOutputCol("document")

# regex tokenizer
regexTokenizer = RegexTokenizer() \
   .setInputCols(["document"]) \
   .setOutputCol("regexToken") \
   .setToLowercase(True) \
   .setPattern("\\s+")

# normalizer
normalizer = Normalizer() \
    .setInputCols(["regexToken"]) \
    .setOutputCol("normalized") \
    .setLowercase(True) \
    .setCleanupPatterns(["""[^\w\d\s]"""]) # remove punctuations (keep alphanumeric chars)

# stopwords clean
stop_words_1 = StopWordsCleaner()\
    .setInputCols("normalized")\
    .setOutputCol("cleanTokens")\
    .setStopWords(stopwords)

# stemmer
stemmer = Stemmer() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("stem")

# stopwords clean
stop_words_2 = StopWordsCleaner()\
    .setInputCols("stem")\
    .setOutputCol("cleanTokens2")\
    .setStopWords(stopwords)

# lemmatizer
lemmatizer =  LemmatizerModel.pretrained()\
    .setInputCols(["cleanTokens2"]) \
    .setOutputCol("lemma") \

# integrate pipeline
prediction_pipeline = Pipeline(
    stages = [
        document,
        regexTokenizer,
        normalizer,
        stop_words_1,
        stemmer,
        stop_words_2,
        lemmatizer
    ]
)

# COMMAND ----------

# set selftext to go over the pipeline
prediction_data = nlp_df.select('selftext')
result = prediction_pipeline.fit(prediction_data).transform(prediction_data)
result.select('lemma.result').show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### text distribution

# COMMAND ----------

# show the results
clean_result = result.select('cleanTokens.result')
clean_result.show(5)

# COMMAND ----------

# get text length and how the results
clean_result = clean_result.withColumn("text_length",f.size(f.col("result")))
clean_result.show(5)

# COMMAND ----------

clean_result = clean_result.withColumn("idx", monotonically_increasing_id())
nlp_df_2 = nlp_df.withColumn("idx", monotonically_increasing_id())
# join the dataframes
nlp_df_2 = nlp_df_2.join(clean_result,["idx"])

# COMMAND ----------

nlp_df_2.write.parquet('/FileStore/ML_preparing')
