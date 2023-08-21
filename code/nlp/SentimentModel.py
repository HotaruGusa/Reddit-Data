# Databricks notebook source
# import package
import pandas as pd
import numpy as np
import json
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
from sparknlp.pretrained import PretrainedPipeline

# COMMAND ----------

MODEL_NAME='classifierdl_use_emotion'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read file

# COMMAND ----------

dummy_nlp=spark.read.parquet("/FileStore/dummy_nlp/dummy.parquet")
dummy_nlp.show(5)

# COMMAND ----------

esport = dummy_nlp.filter(f.col('Not_esport')==0).select("CBLOL","LCK","LCL","LCO","LCS","LEC","LJL","LLA","LPL","PCS","TCL","VCS","WORLDS")

# COMMAND ----------

esport.agg({'WORLDS': 'sum','LCK':'sum','LPL':'sum',
        'CBLOL':'sum','LCL':'sum','LCO':'sum',
         'LCS':'sum','LEC':'sum','LJL':'sum','LLA':'sum','PCS':'sum','VCS':'sum'}).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get information for each esport team

# COMMAND ----------

lck = dummy_nlp.filter(f.col('LCK')==1).select("selftext")
lpl = dummy_nlp.filter(f.col('LPL')==1).select("selftext")
lcs = dummy_nlp.filter(f.col('LCS')==1).select("selftext")
lec = dummy_nlp.filter(f.col('LEC')==1).select("selftext")

# COMMAND ----------

# create the pipeline
documentAssembler = DocumentAssembler()\
    .setInputCol("selftext")\
    .setOutputCol("document")
    
use = UniversalSentenceEncoder.pretrained(name="tfhub_use", lang="en")\
 .setInputCols(["document"])\
 .setOutputCol("sentence_embeddings")


sentimentdl = ClassifierDLModel.pretrained(name=MODEL_NAME)\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("sentiment")

nlpPipeline = Pipeline(
      stages = [
          documentAssembler,
          use,
          sentimentdl
      ])


# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit to pipeline

# COMMAND ----------

pipelineModel_lck = nlpPipeline.fit(lck)
result_lck = pipelineModel_lck.transform(lck)
pipelineModel_lpl = nlpPipeline.fit(lpl)
result_lpl = pipelineModel_lpl.transform(lpl)
pipelineModel_lcs = nlpPipeline.fit(lcs)
result_lcs = pipelineModel_lcs.transform(lcs)
pipelineModel_lec = nlpPipeline.fit(lec)
result_lec = pipelineModel_lec.transform(lec)

# COMMAND ----------

result_lck

# COMMAND ----------

# MAGIC %md
# MAGIC ### Count sentiment situation

# COMMAND ----------

lck_py = result_lck.select(f.explode('sentiment.result').alias("sentiment")).groupBy(f.col('sentiment')).count()
lck_py.show()

# COMMAND ----------

lpl_py = result_lpl.select(f.explode('sentiment.result').alias("sentiment")).groupBy(f.col('sentiment')).count()
lpl_py.show()

# COMMAND ----------

lcs_py = result_lcs.select(f.explode('sentiment.result').alias("sentiment")).groupBy(f.col('sentiment')).count()
lcs_py.show()

# COMMAND ----------

lec_py = result_lec.select(f.explode('sentiment.result').alias("sentiment")).groupBy(f.col('sentiment')).count()
lec_py.show()

# COMMAND ----------

# create a specific label for each esport team
lck_py = lck_py.withColumn("esport", f.lit("LCK"))
lck_py = lck_py.select("esport", "sentiment", "count")
lck_py.show()

# COMMAND ----------

# create labels to each table
lpl_py = lpl_py.withColumn("esport", f.lit("LPL"))
lpl_py = lpl_py.select("esport", "sentiment", "count")

lcs_py = lcs_py.withColumn("esport", f.lit("LCS"))
lcs_py = lcs_py.select("esport", "sentiment", "count")

lec_py = lec_py.withColumn("esport", f.lit("LEC"))
lec_py = lec_py.select("esport", "sentiment", "count")

# COMMAND ----------

# transfer to pandas
lck_df = lck_py.toPandas()
lpl_df = lpl_py.toPandas()
lcs_df = lcs_py.toPandas()
lec_df = lec_py.toPandas()

# COMMAND ----------

lck_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save to CSV

# COMMAND ----------

lck_df.to_csv('../../data/csv/lck_sentiment.csv')
lpl_df.to_csv('../../data/csv/lpl_sentiment.csv')
lcs_df.to_csv('../../data/csv/lcs_sentiment.csv')
lec_df.to_csv('../../data/csv/lec_sentiment.csv')


# COMMAND ----------

# MAGIC %md
# MAGIC ### merge data

# COMMAND ----------

merge1 = lck_py.unionByName(lpl_py)
merge2 = merge1.unionByName(lcs_py)
merge3 = merge2.unionByName(lec_py)
merge3.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Plot pie chart

# COMMAND ----------

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

# COMMAND ----------

LCK_logo =Image.open("sentiment_img/lck.png")
LPL_logo =Image.open("sentiment_img/lpl.png")
LCS_logo =Image.open("sentiment_img/lcs.png")
LEC_logo =Image.open("sentiment_img/lec.png")

# COMMAND ----------

# plot color
color = ['#FFD73B','#EB7C79','#85D2D0','#887BB0']
fig = make_subplots(rows=2, cols=2, specs=[[{'type':'domain'},
                                            {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]],
                    vertical_spacing = 0.01)

# add trace
fig.add_trace(go.Pie(labels=lck_df['sentiment'].tolist(), values=lck_df['count'].tolist(), name="LCK (Korea)",marker_colors=color),
              1, 1)
fig.add_trace(go.Pie(labels=lpl_df['sentiment'].tolist(), values=lpl_df['count'].tolist(), name="LPL (China)"),
              1, 2)
fig.add_trace(go.Pie(labels=lcs_df['sentiment'].tolist(), values=lcs_df['count'].tolist(), name="LCS (North America)"),
              2, 1)
fig.add_trace(go.Pie(labels=lec_df['sentiment'].tolist(), values=lec_df['count'].tolist(), name="LEC (Europe)"),
              2, 2)
fig.update_traces(hole=.5, hoverinfo="label+percent+name")

# get image for each subplots
fig.add_layout_image(
        dict(
            source=LCK_logo,
            xref="x",
            yref="y",
            x=0,
            y=3.05,
            sizex=1.15,
            sizey=1.15,
            ),
)
fig.add_layout_image(
        dict(
            source = LPL_logo,
            xref="x",
            yref="y",
            x=3.8,
            y=3.15,
            sizex=1.2,
            sizey=1.2,
            ),
)
fig.add_layout_image(
        dict(
            source=LCS_logo,
            xref="x",
            yref="y",
            x=0,
            y=0.6,
            sizex=1.2,
            sizey=1.2,
            ),
)
fig.add_layout_image(
        dict(
            source=LEC_logo,
            xref="x",
            yref="y",
            x=3.7,
            y=0.8,
            sizex=1.5,
            sizey=1.5,
            ),
)

fig.update_layout(height=800, width=800,template="plotly_white",title_text="Sentiment Analysis on Different Divisions",title_x=0.5)
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.update_annotations(yshift=-310)

fig.show()
fig.write_html("../../data/plots/esport_sentiment_situation.html")


# COMMAND ----------

df1 =dummy_nlp.filter(f.col('No_monster')==0).filter(f.col('Not_esport')==1)
df1.agg({'creeper': 'sum','drowned':'sum','enderman':'sum',
        'skeleton':'sum','slime':'sum','spider':'sum',
         'witch':'sum','zombie':'sum'}).show()
