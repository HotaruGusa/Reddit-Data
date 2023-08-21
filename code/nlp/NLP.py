# Databricks notebook source
# import wordcloud
!pip install wordcloud

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
# MAGIC ### NLP Pipeline

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

# given a index for each text
clean_result = clean_result.withColumn("idx", monotonically_increasing_id())
nlp_df_2 = nlp_df.select('subreddit').withColumn("idx", monotonically_increasing_id())

# COMMAND ----------

# show the results
clean_result.show(5)

# COMMAND ----------

# join the dataframes
text_distribution = nlp_df_2.join(clean_result,["idx"])
text_distribution.show(10)

# COMMAND ----------

# select the text length and change to dataframe
distribution_pd = text_distribution.select('subreddit','text_length').toPandas()
distribution_pd.head()

# COMMAND ----------

# take log of the text_length
lst = []
for i in range(len(distribution_pd)):
    temp = float(distribution_pd.iloc[i]["text_length"])
    if temp > 0:
        lst.append(np.log10(temp))
    else:
        lst.append(0)       
distribution_pd['text_length_log'] = lst
distribution_pd

# COMMAND ----------

sample_textlen = distribution_pd.sample(n = 10000)

# COMMAND ----------

# graph the log of the text length versus the count
sns.histplot(data = sample_textlen[(sample_textlen['subreddit']=='leagueoflegends')],
             x="text_length_log",
             color="#458B74",
             label="leagueoflegends",
             multiple= "stack",
             alpha = 0.5)
sns.histplot(data = sample_textlen[(sample_textlen['subreddit']=='Minecraft')],
             x="text_length_log",
             color="#007ACC",
             label="Minecraft",
             multiple= "stack",
             alpha = 0.5)
plt.xlabel('Log Text Length',weight='bold').set_fontsize('12')
plt.ylabel('Count', weight='bold').set_fontsize('12')
plt.title('Distribution of Text Length on Each Subreddit',weight='bold',y=1.12).set_fontsize('14')
plt.legend(loc='upper center',frameon=False,ncol=2,bbox_to_anchor=(.5, 1.1))
sns.despine(left=True)

plt.text(-0.5, -130, 'Figure 1: Distribution of posts body length after removing stop words.',
         #ha='center',
        color="#a2a2a2",
         fontsize=10)
plt.text(-0.5, -160, 'Most of the posts are in the length range of 10-50.',
         #ha='center',
        color="#a2a2a2",
         fontsize=10)

# save the graphs
plt.savefig('../../data/plots/Text_length.png', dpi=100, bbox_inches='tight')
plt.show()

# COMMAND ----------

# most common words
MC_common = text_distribution \
    .filter(col('subreddit') == 'Minecraft')\
    .select(explode("result").alias("words_exploded")) \
    .groupBy("words_exploded") \
    .count()\
    .orderBy(col('count').desc())
MC_common

# COMMAND ----------

# transfer to the pandas
MC_common = MC_common.toPandas()
MC_common[:20]

# COMMAND ----------

# lol most common words
LOL_common = text_distribution \
    .filter(col('subreddit') == 'leagueoflegends')\
    .select(explode("result").alias("words_exploded")) \
    .groupBy("words_exploded") \
    .count()\
    .orderBy(col('count').desc())
LOL_common.show()

# COMMAND ----------

# transfer to pandas
LOL_common = LOL_common.toPandas()
LOL_common[:20]

# COMMAND ----------

# save the data
MC_common[:20].to_csv('../../data/csv/MC_common.csv',index=False)
LOL_common[:20].to_csv('../../data/csv/LOL_common.csv',index=False)

# COMMAND ----------

# read the saving csv
LOL_common = pd.read_csv('../../data/csv/LOL_common.csv')
MC_common = pd.read_csv('../../data/csv/MC_common.csv')

# COMMAND ----------

# add a column of LOL and MC
MC_common['subreddit'] = 'Minecraft'
LOL_common['subreddit'] = 'League of Legends'
MC_common['label']= MC_common['words_exploded']+' ('+MC_common['count'].astype(str)+')'
LOL_common['label']= LOL_common['words_exploded']+' ('+LOL_common['count'].astype(str)+')'
common_word = pd.concat([LOL_common,MC_common ])

# COMMAND ----------

# settings for graphs underbelow
ANGLES = np.linspace(0, 2 * np.pi, len(common_word), endpoint=False)
VALUES = ((common_word["count"].values)/900).round(0)
LABELS = common_word["words_exploded"].values
LABELS_2 = common_word["label"].values
COUNT = common_word["count"].values

# COMMAND ----------

def get_label_rotation(angle, offset):
    # rotation must be specified in degrees
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"
    return rotation, alignment

def add_labels(angles, values, labels, offset, ax):
    # this is the space between the end of the bar and the label
    padding = 4
    # iterate over angles, values, and labels, to add all of them.
    for angle, value, label, in zip(angles, values, labels):
        angle = angle
        # Obtain text rotation and alignment
        rotation, alignment = get_label_rotation(angle, offset)
        # And finally add the text
        ax.text(
            x=angle, 
            y=value + padding, 
            s=label, 
            ha=alignment, 
            va="center", 
            rotation=rotation, 
            rotation_mode="anchor"
        ) 

# COMMAND ----------

# grab the group values
GROUP = common_word["subreddit"].values

# add three empty bars to the end of each group
PAD = 3
ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))
ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
WIDTH = (2 * np.pi) / len(ANGLES)

# obtaining the right indexes is now a little more complicated
offset = 0
IDXS = []
GROUPS_SIZE = [20,20]
for size in GROUPS_SIZE:
    IDXS += list(range(offset + PAD, offset + size + PAD))
    offset += size + PAD

# same layout as above
fig, ax = plt.subplots(figsize=(20, 10), subplot_kw={"projection": "polar"})

ax.set_theta_offset(offset)
ax.set_ylim(-100, 100)
ax.set_frame_on(False)
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_xticks([])
ax.set_yticks([])

# use different colors for each group!
GROUPS_SIZE = [20,20]
COLORS = ['#458B74']*20+['#007ACC']*20

# and finally add the bars. 
# note again the `ANGLES[IDXS]` to drop some angles that leave the space between bars.
ax.bar(
    ANGLES[IDXS], VALUES, width=WIDTH, color=COLORS, 
    edgecolor="white", linewidth=2
)

add_labels(ANGLES[IDXS], VALUES, LABELS_2, offset, ax)
#fig.suptitle('Most Common Words In Each Subreddit', fontsize=20, x = 0.5, y = 0.95)

offset = 0 
for group, size in zip(["LOL", "MC"], GROUPS_SIZE):
    # add line below bars
    x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=50)
    ax.plot(x1, [-5] * 50, color="#333333")
    
    # add text to indicate group
    ax.text(
        np.mean(x1), -20, group, color="#333333", fontsize=14, 
        fontweight="bold", ha="center", va="center"
    )
    x2 = np.linspace(ANGLES[offset], ANGLES[offset + PAD - 1], num=50)
    ax.plot(x2, [20] * 50, color="#bebebe", lw=0.8)
    ax.plot(x2, [40] * 50, color="#bebebe", lw=0.8)
    offset += size + PAD
    
fig.text(
    0.3, 0.9,
    "What are the most common words on each subreddit?", 
    fontsize=20, 
    #color="white",
    fontweight="bold")
# Add subtitle
fig.text(
    0.3, 0.865,
    "Counting the word frequencies of posts", 
    #color="white",
    fontsize=16)
# Add caption
fig.text(
    0.3, 0.07,
    "Figure 2: The Top 20 most common words on League of Legends and Minecraft subreddits,", 
    color="#a2a2a2", 
    fontsize=14)
fig.text(
    0.3, 0.05,
    "the number in bracket is the count of that word.", 
    color="#a2a2a2", 
    fontsize=14)
fig.savefig('../../data/plots/common_word.png', dpi=100, bbox_inches='tight')

# COMMAND ----------

# MAGIC %md
# MAGIC ###  Term Frequency–Inverse Document Frequency (TF-IDF)

# COMMAND ----------

# group by and flatten the results
sample = text_distribution
s1 = sample.groupBy('subreddit').agg(f.collect_list('result').alias('result'))
s2 = s1.withColumn("flatten_array", f.flatten("result"))

# COMMAND ----------

# get the TF of words
hashingTF = HashingTF(inputCol="flatten_array", outputCol="rawFeatures")
featurizedData = hashingTF.transform(s2)

# COMMAND ----------

# get the IDF of words
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# COMMAND ----------

# get the schema
rescaledData.printSchema()

# COMMAND ----------

# show the results
rescaledData.select('subreddit','flatten_array','features').show(5)

# COMMAND ----------

# hashing TF and transform the ndf
ndf = sample.select('subreddit',f.explode('result').name('expwords')).withColumn('flatten_array',f.array('expwords'))
hashudf = f.udf(lambda vector : vector.indices.tolist()[0],StringType())
wordtf = hashingTF.transform(ndf).withColumn('wordhash',hashudf(f.col('rawFeatures')))

# COMMAND ----------

# show the results
wordtf.show(10)

# COMMAND ----------

# flatten output features column to get indices and value
udf1 = f.udf(lambda vec : dict(zip(vec.indices.tolist(),vec.values.tolist())),MapType(StringType(),StringType()))
valuedf = rescaledData.select('subreddit', f.explode(udf1(f.col('features'))).name('wordhash','value'))
valuedf.show()

# COMMAND ----------

# output the TFIDF and get the top words
w = Window.partitionBy('subreddit').orderBy(f.desc('value'))
valuedf = valuedf.withColumn('rank',f.rank().over(w)).where(f.col('rank')<=10)
TFIDF = valuedf.join(wordtf,['subreddit','wordhash'])\
    .groupby('subreddit')\
    .agg(f.sort_array(f.collect_list(f.struct(f.col('value'),f.col('expwords'))),asc=False).name('topn'))

# COMMAND ----------

# show the results
TFIDF.show()

# COMMAND ----------

#  flatten map of the TFIDF
TFIDF_scores = TFIDF.select('topn').rdd.flatMap(lambda x: x).collect()

# COMMAND ----------

# get the top important words
LOL_important_words ={}
for i in TFIDF_scores[1]:
    LOL_important_words[i['expwords']]=i['value']
MC_important_words = {}
for i in TFIDF_scores[0]:
    MC_important_words[i['expwords']]=i['value']

# COMMAND ----------

# check the LOL important words
LOL_important_words

# COMMAND ----------

# get the important words of MC and LOL
MC_important_words = {k: v for k, v in sorted(MC_important_words.items(), key=lambda item: item[1],reverse=True)}
LOL_important_words = {k: v for k, v in sorted(LOL_important_words.items(), key=lambda item: item[1],reverse=True)}

# COMMAND ----------

# filter out the strange words that very long
MC_important_words = {k:MC_important_words[k] for k in MC_important_words if len(k)<15}
LOL_important_words = {k:LOL_important_words[k] for k in LOL_important_words if len(k)<15}

# COMMAND ----------

# filter out the strange words that not numbers
MC_important_words = {k:MC_important_words[k] for k in MC_important_words if not re.search("\d+",k)}
LOL_important_words = {k:LOL_important_words[k] for k in LOL_important_words if not re.search("\d+",k)}

# COMMAND ----------

# check the cleaning words of LOL
LOL_important_words

# COMMAND ----------

# check the cleaning words of MC
MC_important_words

# COMMAND ----------

# get TF-IDF top words
MC_TFIDF = pd.DataFrame({'word':list(MC_important_words.keys()),
	       'score':list(MC_important_words.values())})

LOL_TFIDF = pd.DataFrame({'word':list(LOL_important_words.keys()),
	       'score':list(LOL_important_words.values())})

# COMMAND ----------

# save the results
MC_TFIDF.to_csv('../../data/csv/MC_TFIDF.csv',index=False)

# COMMAND ----------

# save the results
LOL_TFIDF

# COMMAND ----------

# round the score
LOL_TFIDF['score'] = LOL_TFIDF['score'].astype(float).round(3)
MC_TFIDF['score'] = MC_TFIDF['score'].astype(float).round(3)

# COMMAND ----------

# save to the csv
LOL_TFIDF = pd.read_csv('../../data/csv/LOL_TFIDF.csv')
MC_TFIDF = pd.read_csv('../../data/csv/MC_TFIDF.csv')

# COMMAND ----------

# check the MC
MC_TFIDF[:10]

# COMMAND ----------

# check the LOL
LOL_TFIDF[:10]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dummy Variable

# COMMAND ----------

# use regex to filter out submission content that related to lol esport division
def esport_dummy(comment):
    region = r'(?i)(lck|lcs|lec|lpl|pcs|vcs|cblol|lco|lcl|ljl|lla|tcl|worlds)'
    l = re.findall(region,comment)
    l = [i.upper() for i in l]
    if len(l) == 0:
        return ['Not_esport']
    else:
        return [*set(l)]

# use user defined function to convert submission selftext to mutiple dummy variables, each one represents one division.
udf_func = udf(esport_dummy,ArrayType(StringType())) 
nlp_df1 = nlp_df.withColumn("region",udf_func(nlp_df.selftext))
nlp_df1 = nlp_df1.withColumn("idx", monotonically_increasing_id())
nlp_df2 = nlp_df1.select(nlp_df1.idx,explode(nlp_df1.region)).\
groupBy('idx').pivot('col').count().na.fill(value=0)
nlp_df1 = nlp_df1.select("idx","selftext").join(nlp_df2,["idx"])

# use regex to find out submission content that mentioned famous monsters in Minecraft
def monster_dummy(comment):
    monster = r'(?i)(zombie|spider|enderman|creeper|skeleton|drowned|witch|slime)'
    l = re.findall(monster,comment)
    l = [i.lower() for i in l]
    if len(l) == 0:
        return ['No_monster']
    else:
        return [*set(l)]

# use user defined function to convert submission selftext to mutiple dummy variables, each one represents one type of monster.
udf_func = udf(monster_dummy,ArrayType(StringType())) 
nlp_df1 = nlp_df1.withColumn("monster",udf_func(nlp_df.selftext))
nlp_df2 = nlp_df1.select(nlp_df1.idx,explode(nlp_df1.monster)).\
groupBy('idx').pivot('col').count().na.fill(value=0)
nlp_df1 = nlp_df1.join(nlp_df2,["idx"])


# filter out submissions that don't contain any information related to esport division/monsters in the Minecraft. Save the data
nlp_df1 = nlp_df1.filter((col("Not_esport")==0)|(col("No_monster")==0))
nlp_df1.write.parquet('/FileStore/dummy_nlp/dummy.parquet')
