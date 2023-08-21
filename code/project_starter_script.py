# Databricks notebook source
# MAGIC %md
# MAGIC # Exploring reddit data using Spark
# MAGIC 
# MAGIC This notebook provides startup code for downloading Reddit data from the Azure Blob Storage bucket specifically setup for this project.   
# MAGIC _<b>Make sure you are running this notebook on a cluster which has the credentials setup to access Azure Blob Storage, otherwise this notebook will not be able to read the data!</b>_
# MAGIC 
# MAGIC _<b>Make sure you are using DataBricks Runtime 11.3 or newer otherwise you will not be able to save any new files in this repository!</b>_
# MAGIC 
# MAGIC The dataset for this notebook is described in [The Pushshift Reddit Dataset](https://arxiv.org/pdf/2001.08435.pdf) paper.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the data
# MAGIC Let us begin with listing the file in the bucket. There are two folders `comments` and `submissions`, we will read both of them into separate Spark dataframes. The raw data is in [parquet format](https://www.databricks.com/glossary/what-is-parquet#:~:text=What%20is%20Parquet%3F,handle%20complex%20data%20in%20bulk.).

# COMMAND ----------

dbutils.fs.ls("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet")

# COMMAND ----------

comments = spark.read.parquet("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/comments")
submissions = spark.read.parquet("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/submissions")

# COMMAND ----------

# MAGIC %md
# MAGIC The `count` operation below will take several minutes, it is show here because it provides an oppurtunity to look at the Spark UI will a computation is in progress. Expand the `Spark Jobs` output in the result cell to see how the job is progressing and then drill down into task details.

# COMMAND ----------

comments_row_count = comments.count()
comment_col_count = len(comments.columns)

# COMMAND ----------

print(f"shape of the comments dataframe is {comments_row_count:,}x{comment_col_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC A glimpse of the data and the schema.

# COMMAND ----------

comments.show()

# COMMAND ----------

comments.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis
# MAGIC 
# MAGIC Here is an example of some EDA. But, before we do that let us setup some local directory structure so that that the results can be checked in into the repo.

# COMMAND ----------

## create a directory called data/plots and data/csv to save generated data
import os
PLOT_DIR = os.path.join("data", "plots")
CSV_DIR = os.path.join("data", "csv")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### How many subreddits are there and which are the more popular ones
# MAGIC 
# MAGIC One of the first questions we can ask is how many subreddits are there and which ones are the top 10 based on the number of submissions.

# COMMAND ----------

from pyspark.sql.functions import col, asc,desc
submissions_by_subreddit = submissions.groupBy("subreddit").count().orderBy(col("count"), ascending=False).collect()

# COMMAND ----------

submissions_by_subreddit

# COMMAND ----------

# MAGIC %md
# MAGIC ## Plot a bar chart for the top 10 most popular by submission count

# COMMAND ----------

top_n = 10
top_n_subreddits = spark.createDataFrame(submissions_by_subreddit[:top_n]).toPandas()
top_n_subreddits

# COMMAND ----------

## Import data visualization packages
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 5]
plt.bar("subreddit", "count", data = top_n_subreddits, color = "blue")
plt.xlabel("subreddit")
plt.ylabel("Submission Count")
plt.title(f"Top {top_n} subreddits by suybmission count")

## Save the plot in the plot dir so that it can be checked in into the repo
plot_fpath = os.path.join(PLOT_DIR, f'top_{top_n}_subreddits.png')

plt.savefig(plot_fpath)
plt.show()

# COMMAND ----------

## save the csv file in the csv dir
fpath = os.path.join(CSV_DIR, "top_n_subreddits.csv")
top_n_subreddits.to_csv(fpath)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Saving data to DBFS
# MAGIC Sometimes we may want to save intermediate data, especially if it is big and took a significant time to generate, in DBFS. The following code shows an example of this. We save submissions on the `memes` subreddit into dbfs so that we can read from it at a later stage.

# COMMAND ----------


submissions_filtered = submissions.filter(submissions.subreddit == "memes")

# COMMAND ----------

submissions_filtered.show()

# COMMAND ----------

# this code saves the data as a csv just for illustration, use parquet for faster analytics operations.
submissions_filtered.select("author_id", "title", "selftext").write.format('com.databricks.spark.csv').save("/FileStore/memes")

# COMMAND ----------

memes = spark.read.csv("/FileStore/memes")
memes.show()

# COMMAND ----------


