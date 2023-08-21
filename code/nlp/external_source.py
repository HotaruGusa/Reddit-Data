# Databricks notebook source
# MAGIC %md
# MAGIC ## External data with Reddit html and Youtube API

# COMMAND ----------

# import libraries
import matplotlib.pyplot as plt
import pandas as pd
import re
import pyspark.sql.functions as f
from pyspark.sql.functions import *
from pyspark.sql.functions import col,sum,avg,max,count
from datetime import datetime, timedelta
import sys
from collections import ChainMap
!pip install wordcloud
from wordcloud import WordCloud, STOPWORDS

# COMMAND ----------

# Read file to local
dbutils.fs.ls("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet")
submissions = spark.read.parquet("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/submissions")

# COMMAND ----------

submission_games = submissions.filter(submissions.subreddit == "Games")

# COMMAND ----------

time_check = submissions.select(col("subreddit"),col("created_utc"))
time_check = time_check.withColumn("submission_created_date",f.from_unixtime(col("created_utc")))
time_check.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Check the data range and select the half year data

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
# MAGIC ### Using external data: the 'Reddit html' to get game list in reddit 
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

top_games.to_csv('../data/csv/top_game.csv', index=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Youtube API
# MAGIC #### Count the total video created by recent date

# COMMAND ----------

!pip install --upgrade google-api-python-client
!pip install --upgrade google-auth google-auth-oauthlib google-auth-httplib2

import os

import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors

# COMMAND ----------

scopes = ["https://www.googleapis.com/auth/youtube.force-ssl"]
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

api_service_name = "youtube"
api_version = "v3"
client_secrets_file = "client_secret_380998963226-bd0mmubs7r18fesr0rv4392n68249887.apps.googleusercontent.com.json"

# Get credentials and create an API client
flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes)
credentials = flow.run_console()
youtube = googleapiclient.discovery.build(api_service_name, api_version, credentials=credentials)

# COMMAND ----------

# total completed videos for league of legends in past 1 day
request = youtube.search().list(
    part="snippet",
    publishedAfter='2022-11-17T00:00:00Z',
    eventType='completed',
    type='video',
    q="league of legends")

lol_response_1_day = request.execute()

print(lol_response_1_day)

# COMMAND ----------

# create a dataframe to hold information
res = pd.DataFrame(columns = ['Name', '1 day', '3 days', '1 week', '1 month'])
lol_info = {'Name': 'league of legends'}
mc_info = {'Name': 'minecraft'}

# COMMAND ----------

lol_info['1 day'] = lol_response_1_day['pageInfo']['totalResults']
lol_response_1_day['pageInfo']['totalResults']

# COMMAND ----------

# total completed videos for league of legends in past 3 days
request = youtube.search().list(
    part="snippet",
    publishedAfter='2022-11-15T00:00:00Z',
    eventType='completed',
    type='video',
    q="league of legends")

lol_response_3_day = request.execute()
lol_info['3 days'] = lol_response_3_day['pageInfo']['totalResults']

# total completed videos for league of legends in past 7 days
request = youtube.search().list(
    part="snippet",
    publishedAfter='2022-11-11T00:00:00Z',
    eventType='completed',
    type='video',
    q="league of legends")

lol_response_7_day = request.execute()
lol_info['1 week'] = lol_response_7_day['pageInfo']['totalResults']

# total completed videos for league of legends in past 1 month
request = youtube.search().list(
    part="snippet",
    publishedAfter='2022-10-18T00:00:00Z',
    eventType='completed',
    type='video',
    q="league of legends")

lol_response_30_day = request.execute()
lol_info['1 month'] = lol_response_30_day['pageInfo']['totalResults']

res = res.append(lol_info, ignore_index=True)
res

# COMMAND ----------

# total completed videos for minecraft in past 1 day
request = youtube.search().list(
    part="snippet",
    publishedAfter='2022-11-17T00:00:00Z',
    eventType='completed',
    type='video',
    q="minecraft")

mc_response_1_day = request.execute()

print(mc_response_1_day)

# COMMAND ----------

mc_info['1 day'] = mc_response_1_day['pageInfo']['totalResults']
mc_response_1_day['pageInfo']['totalResults']

# total completed videos for minecraft in past 3 days
request = youtube.search().list(
    part="snippet",
    publishedAfter='2022-11-15T00:00:00Z',
    eventType='completed',
    type='video',
    q="minecraft")

mc_response_3_day = request.execute()
mc_info['3 days'] = mc_response_3_day['pageInfo']['totalResults']

# total completed videos for minecraft in past 7 days
request = youtube.search().list(
    part="snippet",
    publishedAfter='2022-11-11T00:00:00Z',
    eventType='completed',
    type='video',
    q="minecraft")

mc_response_7_day = request.execute()
mc_info['1 week'] = mc_response_7_day['pageInfo']['totalResults']

# total completed videos for minecraft in past 1 month
request = youtube.search().list(
    part="snippet",
    publishedAfter='2022-10-18T00:00:00Z',
    eventType='completed',
    type='video',
    q="minecraft")

mc_response_30_day = request.execute()
mc_info['1 month'] = mc_response_30_day['pageInfo']['totalResults']

res = res.append(mc_info, ignore_index=True)
res

# COMMAND ----------


