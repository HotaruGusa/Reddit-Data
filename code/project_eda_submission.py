# Databricks notebook source
# MAGIC %md
# MAGIC ## Subreddit Project Submission Data Cleaning & EDA
# MAGIC ### Data Cleaning

# COMMAND ----------

# import libraries
!pip install wordcloud
import pandas as pd
import re
import pyspark.sql.functions as f
from pyspark.sql.functions import *
from pyspark.sql.functions import col,sum,avg,max,count
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import subprocess
import sys
from collections import ChainMap
from wordcloud import WordCloud, STOPWORDS
import plotly.graph_objects as go
from PIL import Image
import numpy as np
from matplotlib.patheffects import withStroke
import plotly.express as px

# COMMAND ----------

# Read file to local
dbutils.fs.ls("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet")
comments = spark.read.parquet("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/comments")
submissions = spark.read.parquet("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/submissions")

# COMMAND ----------

# print out the Schema
comments.printSchema()

# COMMAND ----------

submissions.printSchema()

# COMMAND ----------

submission_games = submissions.filter(submissions.subreddit == "Games")
comments_games = comments.filter(comments.subreddit == "Games")

# COMMAND ----------

submission_games.count()

# COMMAND ----------

comments_games.count()

# COMMAND ----------

time_check = submissions.select(col("subreddit"),col("created_utc"))
time_check = time_check.withColumn("submission_created_date",f.from_unixtime(col("created_utc")))
time_check.show(5)


# COMMAND ----------

# MAGIC %md
# MAGIC ##### A summary table for the newest post date and latest post date for our dataset

# COMMAND ----------

# get the newest/lastest date of dataset
max_min_time = time_check.agg(max("submission_created_date").alias("Newest_date"), min("submission_created_date").alias("Latest_date"))
max_min_time.show()
max_min_time = max_min_time.cache()

# COMMAND ----------

row_list = max_min_time.collect()
last = row_list[0].__getitem__('Newest_date')

# COMMAND ----------

# only calc data in the past 180 days(half year)
last = datetime.strptime(last, '%Y-%m-%d %H:%M:%S')
last_half_year = last-timedelta(days=180)
print(last)
print(last_half_year)

# COMMAND ----------

# filter data
half_year = time_check.filter(time_check.submission_created_date.between(last_half_year,last))
half_year.show()

# COMMAND ----------

# count how many data we have
overview_half_year = half_year.groupBy("subreddit").count().orderBy(col("count"), ascending=False)
overview_half_year.show(10)

# COMMAND ----------

overview_pd = overview_half_year.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Top 10 subreddits for games
# MAGIC Count the number of games people discussed in the past half year

# COMMAND ----------

# get game list from reddit offical site
all_game = []
file = open('list-sorted-by-subscribers - gaming.html', 'r')
text = file.read()
inf = re.findall(r'<td><a href.*</td>', text)
for i in inf:
    inf2 = re.findall("/r/(\w+)<", i)
    all_game.append(inf2[0])
print(all_game[:10])

# COMMAND ----------

# get top 10 games in games subreddit
overview_pd = overview_pd.rename(columns={"count": "total_reddits"})
top_games = pd.DataFrame(columns=['name','total_reddits'])
for i in range(len(overview_pd.subreddit)):
    if overview_pd.subreddit[i] in all_game:
        df = {'name': overview_pd.subreddit[i], 'total_reddits': overview_pd.total_reddits[i]}
        top_games = top_games.append(df, ignore_index = True)
    if len(top_games) == 10:
        break
print(top_games)

# COMMAND ----------

top_games.to_csv('data/csv/top_game.csv', index=False)


# COMMAND ----------

top_games.head()

# COMMAND ----------

top_games['total_reddits'] = top_games['total_reddits']/1000

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data visualization

# COMMAND ----------

top_game = pd.read_csv('../data/csv/top_game.csv')
top_game['total_reddits'] = top_game['total_reddits']/1000
y = [i * 0.9 for i in range(len(top_game))]

# COMMAND ----------

fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(y, top_game["total_reddits"].tolist(), height=0.55, align="edge", color="#076fa2")

ax.xaxis.set_ticks([i * 20 for i in range(0, 9)])
ax.xaxis.set_ticklabels([i * 20 for i in range(0, 9)], size=16,fontweight=100)
ax.xaxis.set_tick_params(labelbottom=False, labeltop=True, length=0)

ax.set_xlim((0, 180))
ax.set_ylim((0, len(top_game) * 0.9 - 0.2))

# Set whether axis ticks and gridlines are above or below most artists.
ax.set_axisbelow(True)
ax.grid(axis = "x", color="#A8BAC4", lw=1.2)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_lw(1.5)

ax.spines["left"].set_capstyle("butt")

# Hide y labels
ax.yaxis.set_visible(False)

PAD = 0.5
for name, count, y_pos in zip(top_game["name"].tolist(),top_game["total_reddits"].tolist(),y):
    x = 0
    color = "white"
    path_effects = None
    if count < 60:
        x = count
        color = "#076fa2"    
        path_effects=[withStroke(linewidth=6, foreground="white")]
    
    ax.text(
        x + PAD, y_pos + 0.5 / 2, name, 
        color=color, fontsize=18, va="center",
        path_effects=path_effects
    ) 
ax.yaxis.set_visible(False)

fig.subplots_adjust(left=0.005, right=1, top=0.8, bottom=0.1)

# Add title
fig.text(
    0, 0.925, "Top 10 Games on Reddit", 
    fontsize=22, fontweight="bold")
# Add subtitle
fig.text(
    0, 0.87, "Number of Posts (thousands) on subreddit, 2022/2 - 2022/8", 
    fontsize=18)

# Add caption
source = "Figure 1: The most popular games on Reddit in the past year."
fig.text(
    0, 0.06, source, color="#a2a2a2", 
    fontsize=14)

# Set facecolor, useful when saving as .png
fig.set_facecolor("white")
fig.savefig('../data/plots/hist2.png', dpi=160, bbox_inches='tight', transparent=True)

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC Create two WordCloud plots for each gaming subreddit

# COMMAND ----------

mc = submissions.filter(submissions.subreddit == "Minecraft")
lol = submissions.filter(submissions.subreddit == "leagueoflegends")

# COMMAND ----------

# word cloud for league of legends 
subprocess.check_call([sys.executable, '-m', 'pip', 'install','wordcloud'])

stopwords = set(STOPWORDS)
# set new stopwords
more_stop = ['mt', 'https', 'c', 'removed', 'deleted']
for i in more_stop:
    stopwords.add(i)
text = lol.select("selftext").rdd.flatMap(lambda x: x).collect()
texts = " ".join(i for i in text)

# COMMAND ----------

mask = np.array(Image.open('t3.png'))
wordcloud = WordCloud(stopwords=stopwords, background_color="black", colormap='Set2', mask=mask, collocations=False).generate(texts)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('wordcloud for league of legends', fontsize=25)
plt.savefig('../data/plots/lol_wordcloud.png', dpi=160, bbox_inches='tight', transparent=True)
plt.show()

# COMMAND ----------

# word cloud for minecraft
text = mc.select("selftext").rdd.flatMap(lambda x: x).collect()
texts = " ".join(i for i in text)

# COMMAND ----------

mask = np.array(Image.open('t2.png'))
wordcloud = WordCloud(stopwords=stopwords, background_color="black", mask=mask, colormap='Set2', collocations=False).generate(texts)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('wordcloud for minecraft', fontsize=25)
plt.axis("off")
plt.savefig('../data/plots/mc_wordcloud.png', dpi=160, bbox_inches='tight', transparent=True)
plt.show()

# COMMAND ----------

# over-18 column situation
over18 = submissions.filter(submissions.subreddit.isin(top_games.name.to_list())).groupBy("subreddit","over_18").count().toPandas()
over18.to_csv('../data/csv/over_18.csv',index=False)

# COMMAND ----------

total_count = submissions.filter(submissions.subreddit.isin(top_games.name.to_list())).groupBy("subreddit").count().toPandas()

total_count.to_csv('../data/csv/total_count.csv',index=False)
total_count.sort_values('subreddit')

# COMMAND ----------

over18[over18['over_18']==False].sort_values('subreddit').reset_index()

# COMMAND ----------

top_games = pd.read_csv('../data/csv/top_game.csv')
top_games.sort_values('name')

# COMMAND ----------

# total count base on 10 games and over-18 column
total_count = pd.read_csv('../data/csv/total_count.csv')
over18 = pd.read_csv('../data/csv/over_18.csv')
total_count = total_count.sort_values('subreddit').reset_index(drop=True)
total_count['health'] = over18[over18['over_18']==False].sort_values("subreddit").reset_index()['count']

# COMMAND ----------

total_count

# COMMAND ----------

total_count['R-18'] = total_count['count'] - total_count['health']
total_count.drop('count', axis=1, inplace=True)
total_count.rename({'health': 'All Age'}, axis=1, inplace=True)

total_count = total_count.melt(id_vars='subreddit', 
        var_name='Content', 
        value_name="Value")

# COMMAND ----------

pxfig = px.sunburst(total_count, path=['subreddit','Content'],values='Value',width=550, height=550,
                    title = "SENSITIVE CONTENT IN DIFFERENT GAME SUBREDDITS"
                   )
pxfig.update_layout(title_x=0.5)
#pxfig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
pxfig.update_traces(texttemplate="%{label}<br>%{percentEntry:.2%}")

color_mapping = {'All Age': "#458B00",
                 'R-18': "red",
                'ClashOfClans':'#FFD700',
                'DestinyTheGame':'#555555',
                'DnD':'pink',
                'FIFA':'#C6E2FF',
                'Minecraft':'#458B74',
                'Terraria':'#8B3626',
                'Warthunder':'#8B8B00',
                'halo':'#848484',
                'leagueoflegends':'#007ACC',
                 'pokemon':'#FF8000'
                }
pxfig.update_traces(marker_colors=[color_mapping[cat] for cat in pxfig.data[-1].labels])
#pxfig.write_html("../data/plots/over_18.html")

# COMMAND ----------

# pie chart for over-18 column
total = total_count['count'].sum()
labels = ["total posts"]+total_count['subreddit'].to_list()+[i+"_health" for i in total_count['subreddit'].to_list()]
parents = [""]+["total posts" for _ in total_count["subreddit"]]+total_count['subreddit'].to_list()
values = [total]+total_count['count'].to_list()+total_count["health"].to_list()

fig = go.Figure(go.Sunburst(
    labels=labels,
    parents=parents,
    values=values,
))


fig.update_layout(
    title={
        'text': "Sensitive content in each game subreddit",
        'y':0,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'bottom'})


fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))
fig.write_html("../data/plots/over_18.html")
fig.show()

# COMMAND ----------

# average score situation base on 10 games and over-18 column

over18_score=submissions.filter(submissions.subreddit.isin(top_games.name.to_list())).groupBy(
    "subreddit","over_18").agg(
    avg("score")).orderBy("subreddit","over_18")

# COMMAND ----------

over18_score.show()

# COMMAND ----------

over18_score.toPandas().to_csv("../data/csv/over18_score.csv",index=False)

# COMMAND ----------

sad = pd.read_csv('../data/csv/over18_score.csv')
print(sad)

# COMMAND ----------

# score situation summary
score_summary = submissions.filter(submissions.subreddit.isin(top_games.name.to_list())).groupby(
    'subreddit').agg(f.mean('score').alias('mean_score'),
                                     f.min('score').alias('min_score'),
                                     f.expr('percentile(score, array(0.25))')[0].alias('Q25'),
                                     f.expr('percentile(score, array(0.50))')[0].alias('median'),
                                     f.expr('percentile(score, array(0.75))')[0].alias('Q75'),
                                     f.max('score').alias('max_score'),
                                     f.count(col('score')).alias('total_number'))
score_summary.show()

# COMMAND ----------

# select interesting column and add new variables to dataset
submission_games = submissions.filter(submissions.subreddit.isin(['Minecraft', 'leagueoflegends']))
submission_games = submission_games.withColumn("submission_created_date",f.from_unixtime(col("created_utc")))
submission_games = submission_games.withColumn('hour',f.hour(submission_games.submission_created_date))
submission_games = submission_games.withColumn("day_of_week", date_format(col("submission_created_date"), "E"))

# COMMAND ----------

cols = ['subreddit', 'author', 'score', 'hide_score', 'over_18', 'is_video', 'gilded', 'distinguished', 'created_utc', 'selftext', 'stickied', 'num_comments', 'disable_comments', 'day_of_week', 'hour']
two_games = submission_games.select(cols)

# COMMAND ----------

two_games.printSchema()

# COMMAND ----------

# check NAs may exist in our selected dataset
checks = ['subreddit', 'author', 'score', 'gilded', 'distinguished', 'created_utc', 'selftext', 'num_comments', 'day_of_week', 'hour']
df = two_games.select([count(when(col(c).contains('None') | \
                            col(c).contains('NULL') | \
                            (col(c) == '' ) | \
                            col(c).isNull(), c 
                           )).alias(c)
                    for c in two_games.columns])
df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC There are nearly 450,000 missing values in the important column "selftext". Since it is the main body of our project in NLP analysis, it is impossible to analyze if this variable is missing. So the rows where selftext is missing will be removed.
# MAGIC Other columns "distinguished", "" and "author" also have missing values. "Author" column only has 47 missing values. Since 47 is a very tiny portion of the whole dataset, dropping the rows will not have a big effect on the whole dataset. "distinguished" column has huge amount of missing values, it might be dropped in future machine learning analysis

# COMMAND ----------

two_games.write.parquet("/FileStore/ML_data")

# COMMAND ----------

# regex to see if people discuss two games in 'Games' subreddit
submission_games = submissions.filter(submissions.subreddit == "Games")
submission_games = submission_games.withColumn("talked",when((col("selftext")
                                .rlike(r'(?i)minecraft|league of legends')) ,'0').otherwise('1'))
submission_games = submission_games.select('id', 'talked')
submission_games.show(10)

# COMMAND ----------

# check the summary condition
submission_games.groupBy("talked").agg(    
    count(when(col("talked") == 0, True)),
    count(when(col("talked") == 1, False))).show()
