# Databricks notebook source
# MAGIC %md
# MAGIC ## Subreddit Project Comment Data Cleaning & EDA
# MAGIC ### Data Cleaning

# COMMAND ----------

# import libraries
from pyspark.sql.functions import to_timestamp,date_format
from pyspark.sql.functions import isnan, when, count, col
from datetime import datetime, timedelta
import pyspark.sql.functions as f
from pyspark.sql.functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math

from flexitext import flexitext
from matplotlib import lines
from matplotlib import patches
from matplotlib.patheffects import withStroke

# COMMAND ----------

!pip install flexitext

# COMMAND ----------

# read file to local
dbutils.fs.ls("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet")
submissions = spark.read.parquet("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/submissions")
comments = spark.read.parquet("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/comments")

# COMMAND ----------

# filter comments of minecraft, leagueoflegends
comment_minecraft = comments.filter(comments.subreddit == "Minecraft")
comment_leagueoflegends = comments.filter(comments.subreddit == "leagueoflegends")

# COMMAND ----------

# check submission's schema
submissions.printSchema()

# COMMAND ----------

# check comment's schema
comment_minecraft.printSchema()

# COMMAND ----------

# select columns for EDA using
comment_minecraft = comment_minecraft.select('score', 'created_utc')
comment_leagueoflegends = comment_leagueoflegends.select('score', 'created_utc')

# COMMAND ----------

# create a new variable column called type with "Minecraft" and "LeagueOfLegends"
comment_minecraft = comment_minecraft.withColumn("type", lit("Minecraft"))
comment_leagueoflegends = comment_leagueoflegends.withColumn("type", lit("LeagueOfLegends"))

# COMMAND ----------

# show the top five rows
comment_minecraft.show(5)

# COMMAND ----------

# union the minecraft and leagueoflegends
full_comment = comment_minecraft.union(comment_leagueoflegends)

# COMMAND ----------

# transfer the UTC time to the YY:MM:DD HH:MM:SS
full_comment = full_comment.withColumn("submission_created_date",f.from_unixtime(col("created_utc")))

# COMMAND ----------

# get the newest/lastest date of dataset
max_min_time = full_comment.agg(max("submission_created_date").alias("Newest_date"), min("submission_created_date").alias("Latest_date"))
max_min_time.show()
max_min_time = max_min_time.cache()
row_list = max_min_time.collect()
last = row_list[0].__getitem__('Newest_date')

# COMMAND ----------

# only calc data in the past 180 days(half year), because of the timeliness
last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S')
last_half_year = last-timedelta(days=180)
print(last)
print(last_half_year)

# COMMAND ----------

# filter out the comments between 180 dyas
halfyear_comment = full_comment.filter(full_comment.submission_created_date.between(last_half_year,last))
halfyear_comment.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Create a new variable "week_day_number"
# MAGIC Convert Creation of from UTC time to YY-MM-DD HH:MM:SS and then find out the day of week

# COMMAND ----------

# change the UTC date to the week day number
halfyear_comment = halfyear_comment.withColumn("week_day_number", date_format(col("submission_created_date"), "E"))

# COMMAND ----------

# get all of the hour and save in to a column
halfyear_comment = halfyear_comment.withColumn('hour',f.hour(halfyear_comment.submission_created_date))
halfyear_comment.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Create a new variable "hour"
# MAGIC Convert Creation of from UTC time to YY-MM-DD HH:MM:SS and then extract hours from datetime

# COMMAND ----------

# count the number of hour and type and save to the "halfyear_active"
halfyear_active = halfyear_comment.groupby('hour','type').count().cache().orderBy(['hour','type'], ascending=True)
halfyear_active.show()

# COMMAND ----------

# transfer to the pandas dataframe
halfyear_active_pd = halfyear_active.toPandas()

# COMMAND ----------

# Save to csv without csv
halfyear_active_pd.to_csv('data/csv/halfyear_active_pd.csv', index=False)    

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data quality check
# MAGIC Check the missing values of each column and number of rows in the filtered dataframe.

# COMMAND ----------

# count the halfyear comment
halfyear_comment.count()

# COMMAND ----------

# check missing values in each column
halfyear_comment.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in halfyear_comment.columns]).show()

# COMMAND ----------

# MAGIC %md
# MAGIC We have approximately 3200000 rows in total and selected columns do not have any missing values.

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Visualization

# COMMAND ----------

halfyear_active_pd = pd.read_csv('../data/csv/halfyear_active_pd.csv')

# COMMAND ----------

halfyear_active_pd.head()

# COMMAND ----------

#halfyear_active_pd[halfyear_active_pd['type']=='Minecraft']['hour']
Hour = halfyear_active_pd[halfyear_active_pd['type']=='LeagueOfLegends']['hour']
MC_count = halfyear_active_pd[halfyear_active_pd['type']=='Minecraft']['count']
LOL_count = halfyear_active_pd[halfyear_active_pd['type']=='LeagueOfLegends']['count']
active_count = [MC_count, LOL_count]

# COMMAND ----------

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(Hour, MC_count, color='#007ACC', marker='o', linestyle='dashdot')
ax.scatter(Hour, MC_count, fc='#007ACC', s=100, lw=1.5, ec="white", zorder=12)
ax.plot(Hour, LOL_count, color='#458B74', marker='s')
ax.scatter(Hour, LOL_count, fc='#458B74', s=100, lw=1.5, ec="white", zorder=12)

# Customize y-axis ticks
ax.yaxis.set_ticks([i * 10000 for i in range(4, 13)])
ax.yaxis.set_ticklabels([i * 10000 for i in range(4, 13)])
ax.yaxis.set_tick_params(labelleft=False, length=0)

# Make gridlines be below most artists.
ax.set_axisbelow(True)

# Add grid lines
ax.grid(axis = "y", color="#A8BAC4", lw=1.2)

# Remove all spines but the one in the bottom
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)

# Customize bottom spine
ax.spines["bottom"].set_lw(1.2)
ax.spines["bottom"].set_capstyle("butt")

# Set custom limits
ax.set_ylim(30000, 130000)
ax.set_xlim(-1, 24.5)




PAD = 35 * 0.01
for label in [i * 10000 for i in range(4, 13)]:
    ax.text(
        26, label + PAD, label, 
        ha="right", va="baseline", fontsize=14,
        fontweight=100
    )

    
# Annotate labels for regions ------------------------------

# Note the path effect must be a list
path_effects = [withStroke(linewidth=10, foreground="white")]

# We create a function to avoid repeating 'ax.text' many times
def add_region_label(x, y, text, color, path_effects, ax):
    ax.text(
        x, y, text, color=color,
        fontfamily="Econ Sans Cnd", fontsize=18, 
        va="center", ha="left", path_effects=path_effects
    ) 
region_labels = [
    {
        "x": 13.5, "y": 70000, "text": "Minecraft", 
        "color": "#007ACC", "path_effects": path_effects},
    {
        "x": 11, "y": 115000, "text": "League Of Legends", 
        "color": '#458B74', "path_effects": []
    }
]    

for label in region_labels:
    add_region_label(**label, ax=ax)
    
    
# Add title -----------------------------------------------

# Add title
fig.text(
    0.12, 0.93, "At What Time is Each Subreddit Most Active?", 
    fontsize=20, fontweight="bold")
# Add subtitle
fig.text(
    0.12, 0.89, "Number of Comments on Subreddit per Hour, 2022/2 - 2022/8", 
    fontsize=15)

# Add caption
fig.text(
    0.12, 0.02,
    "Figure 5: The most popular times to League of Legends and Minecraft subreddits.", color="#a2a2a2", 
    fontsize=11)
fig.text(
    0.12, -0.01,
    "Users in both subreddits are most active between the times of 3 PM to about 8 PM.", color="#a2a2a2", 
    fontsize=11)
fig.savefig('../data/plots/most_active.png', dpi=100, bbox_inches='tight')

# COMMAND ----------

# transfer to the comment pandas
halfyear_comment_pd = halfyear_comment.toPandas()

# COMMAND ----------

# show the results
halfyear_comment_pd

# COMMAND ----------

# limit the sample to 10000 to draw the graph
halfyear_comment_limited_pd = halfyear_comment_pd.sample(10000)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create a new variable "log_score"

# COMMAND ----------

# doing the log tranformation and append to a list
lst = []
for i in range(len(halfyear_comment_limited_pd)):
    temp = float(halfyear_comment_limited_pd.iloc[i]["score"])
    if temp > 0:
        lst.append(math.log(temp))
    elif temp < 0:
        lst.append(-math.log(-temp))
    else:
        lst.append(0)

# COMMAND ----------

# append to a new list
halfyear_comment_limited_pd['log_score'] = lst

# COMMAND ----------

# check the 10000 row dataframe
halfyear_comment_limited_pd

# COMMAND ----------

# save to the parquet
halfyear_comment_limited_pd.to_parquet('/dbfs/FileStore/comments_eda/comments_eda.parquet')

# halfyear_comment_limited_pd.to_csv('data/csv/halfyear_comment_limited_pd.csv', index=False)

# COMMAND ----------

dbutils.fs.ls('/dbfs/FileStore/comments_eda/comments_eda.parquet')

# COMMAND ----------

halfyear_comment_limited_pd = spark.read.parquet("/FileStore/comments_eda/comments_eda.parquet")
halfyear_comment_limited_pd = halfyear_comment_limited_pd.toPandas()

# COMMAND ----------

 #halfyear_comment_limited_pd.head(5)

# COMMAND ----------

# print the violin plot with the comparing column minecraft and league of legends
sns.violinplot(data=halfyear_comment_limited_pd,  x="week_day_number", y="log_score", hue="type",
           order=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
           palette="Paired")
plt.xlabel('Day of Week',weight='bold').set_fontsize('12')
plt.ylabel('Log Comment Score', weight='bold').set_fontsize('12')
plt.title('Relationship Between Comments Score and Weekdays',weight='bold',y=1.12).set_fontsize('14')
plt.legend(loc='upper center',frameon=False,ncol=2,bbox_to_anchor=(.5, 1.1))
plt.text(-1.4, -11, 'Figure 6: The violin plot is measuring the comments score and the creation',
         #ha='center',
        color="#a2a2a2",
         fontsize=10)
plt.text(-1.4, -12, 'time (day of the weekend) comparing the League of Legends and Minecraft.',
         #ha='center',
        color="#a2a2a2",
         fontsize=10)
plt.savefig('../data/plots/RelationshipBetweenLogCommentsScoreAndWeekDays.png', dpi=100, bbox_inches='tight')
sns.despine(left=True)


