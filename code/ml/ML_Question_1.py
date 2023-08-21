# Databricks notebook source
# MAGIC %md
# MAGIC ### Import Package

# COMMAND ----------

# import wordcloud
!pip install wordcloud

# import gensim
!pip install gensim

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
from pyspark.sql.types import ArrayType,StringType,IntegerType
from gensim.models import Word2Vec
import gensim
from gensim.utils import simple_preprocess
import requests
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.classification import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense, LSTM,Flatten
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn import metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read File

# COMMAND ----------

# read from submission data
ml_df = spark.read.parquet("/FileStore/ML_preparing")
ml_df.printSchema()

# COMMAND ----------

# start pyspark
spark = sparknlp.start()
print("Spark NLP version", sparknlp.version())
print("Apache Spark version:", spark.version)

# COMMAND ----------

ml_df = ml_df.select('subreddit','result')
ml_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clean Data

# COMMAND ----------

# drop words which has extremely long length
ml = ml_df.withColumn("text", f.expr("filter(result, x -> not(length(x) >= 15))")).drop("result")
ml.show()

# COMMAND ----------

# change labels to 0 and 1
stateDic={'leagueoflegends':0,'Minecraft':1}
ml_label=ml.rdd.map(lambda x: 
    (stateDic[x.subreddit],x.text) 
    ).toDF(["label","text"])
ml_label.show()

# COMMAND ----------

# bonus:
# pipeline: change text to vectors
count = CountVectorizer(inputCol="text", outputCol="rawFeatures")
idf = IDF(inputCol="rawFeatures", outputCol="features")
pipeline = Pipeline(stages=[count, idf])

# COMMAND ----------

pipelineModel = pipeline.fit(ml_label)
rescaledData = pipelineModel.transform(ml_label)
rescaledData.select("label", "features").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split to Train/Test Data

# COMMAND ----------

rescaledData.na.drop()
rescaledData = rescaledData.withColumn("size", f.size(f.col('text')))
rescaledData = rescaledData.filter(f.col("size") >= 1)

from pyspark.sql.window import Window as W
windowSpec = W.orderBy("idx")
rescaledData_idx = rescaledData.withColumn("idx", monotonically_increasing_id())
rescaledData_idx = rescaledData_idx.withColumn("idx", f.row_number().over(windowSpec))
rescaledData_idx.show()

# COMMAND ----------

trainDF_temp, testDF_temp = rescaledData_idx.select("idx").randomSplit(weights = [0.80, 0.20], seed = 12)
trainDF_temp = rescaledData_idx.join(trainDF_temp, on="idx", how="inner")
testDF_temp = rescaledData_idx.join(testDF_temp, on="idx", how="inner")

trainDF = trainDF_temp.select("label", "features")
testDF = testDF_temp.select("label", "features")

trainDF.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit to Logistic Regression

# COMMAND ----------

lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=1)
lrModel = lr.fit(trainDF)

# COMMAND ----------



# COMMAND ----------

# Plot ROC Curve
trainingSummary = lrModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regress ROC Curve')
plt.show()
plt.savefig('../../data/plots/LR_ROC.png', dpi=160, bbox_inches='tight')

print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check Test Data

# COMMAND ----------

predictions = lrModel.transform(testDF)
predictions.show(10)

# COMMAND ----------

# confusion matrix
l = testDF.select('label').rdd.flatMap(lambda x: x).collect()
predict = lrModel.transform(testDF).select('prediction').rdd.flatMap(lambda x: x).collect()
predict = [int(i) for i in predict]
m = confusion_matrix(l, predict,labels=[1,0])
disp = ConfusionMatrixDisplay(confusion_matrix=m,
                              display_labels=[0,1]).plot()
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# COMMAND ----------

metrics.log_loss(l, predict)

# COMMAND ----------

metrics.accuracy_score(l,predict)

# COMMAND ----------

metrics.roc_auc_score(l, predict)

# COMMAND ----------

# get accuracy
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
predictionAndTarget = lrModel.transform(testDF).select('label','prediction')
acc = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "accuracy"})
acc

# COMMAND ----------

lrModel.save('/dbfs/FileStore/lrML_model/lrmodel')

# COMMAND ----------

# MAGIC %md
# MAGIC ### LSTM Model

# COMMAND ----------

# Use word2vec model, creat word-embedding for all the data
sentence = ml.select('text').rdd.flatMap(lambda x: x).collect()
model = Word2Vec(sentences=sentence, vector_size=10, window=5, min_count=3, workers=4)

# Use the word-embedding tranform each word to a vector that has size of 10, and drop word that reraly appear in the text.

del sentence
ml_pd = ml.toPandas()
def f(x):
    l = []
    for i in x:
        try:
            l.append(model.wv[i])
        except KeyError:
            pass
    return np.array(l)
ml_pd['vec']= ml_pd['text'].apply(f)


# COMMAND ----------

train_idx = trainDF_temp.select('idx').rdd.flatMap(lambda x: x).collect()
test_idx =  testDF_temp.select('idx').rdd.flatMap(lambda x: x).collect()

# COMMAND ----------

#convert labels to 1 and 0 
labels=ml_pd['subreddit'].apply(lambda f:[1,0] if f == 'Minecraft' else [0,1]).to_list()
labels = np.array([np.array(xi,dtype='float32') for xi in labels])
#Padding the traning data so each sentence has the same length
ready_data = pad_sequences(ml_pd['vec'].to_list(),maxlen=50,padding='post',dtype='float32')
#split train test set
x_train = ready_data[train_idx]
x_test = ready_data[test_idx]
y_train = labels[train_idx]
y_test = labels[test_idx]

# COMMAND ----------

# Create a LSTM model using keras, which consist a LSTM layer and a fully connected layer.
model = Sequential()
model.add(LSTM(32, dropout = 0.3, recurrent_dropout = 0.3))
model.add(Dense(16, activation = 'relu'))
model.add(Flatten())
model.add(Dense(2, activation = 'softmax'))
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.RMSprop(),
    metrics=['accuracy']
)

# COMMAND ----------

# Train the LSTM model
history = model.fit(x_train, y_train, epochs=10,batch_size=64, validation_split=0.2,verbose=1)

# COMMAND ----------

#Save the model
model.save('/dbfs/FileStore/ML_model/LSTM')

# COMMAND ----------

#Run the model on the test set, the result indicates there's no overfit
model.evaluate(x_test, y_test, batch_size=128)

# COMMAND ----------

#plot to show the training process
hd = history.history
loss = hd['accuracy']
val_loss = hd['val_accuracy']
epochs = range(1,len(loss)+1)
plt.plot(epochs, loss,'bo',label='Training Accuracy')
plt.plot(epochs, val_loss,'b',label='Validation Accuracy')
plt.legend()
plt.title('LSTM')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# COMMAND ----------

# confusion matrix
predict = model.predict(x_test)
m = confusion_matrix(np.argmax(y_test,axis=-1), np.argmax(predict,axis=-1),labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=m,
                              display_labels=[0,1]).plot()
plt.title("LSTM Confusion Matrix")
plt.show()

# COMMAND ----------

predict = model.predict(x_train)
fpr, tpr, _ = metrics.roc_curve(np.argmax(y_train,axis=-1), np.argmax(predict,axis=-1))

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('LSTM ROC Curve')
plt.show()

# COMMAND ----------


metrics.roc_auc_score(np.argmax(y_train,axis=-1), np.argmax(predict,axis=-1))

# COMMAND ----------

metrics.log_loss(np.argmax(y_test,axis=-1), np.argmax(predict,axis=-1))
