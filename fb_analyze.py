# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy
import json
import os
import glob
import re
import seaborn as sns
import plotly.express as px
from matplotlib import pyplot as plt
from matplotlib import rcParams
from datetime import datetime, timedelta


# %%
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Roboto']


# %%
def parse_single_json(path, name):
    # Open Json
    try:
        with open(path) as json_data:
            data = json.load(json_data)
            df = pd.DataFrame.from_dict(data['messages'])
            df = df[df['sender_name'] == name]
            return df
    except:
        pass
        print("JSON Loaded")
    # Create Dataframe from JSON


# %%
os.getcwdb()


# %%
def get_paths():
    path_arr = {}
    cwd = str(os.getcwdb())[2:-1]
    for root, dirs, files in os.walk(cwd):
        for file in files:
            if file.endswith(".json"):
                #  path_arr.append(os.path.join(root, file))
                path_arr[os.path.join(root, file)] = os.path.join(root, file).split('\\')[-2][0:-11]
    return(path_arr)


# %%
name = "Ali Adnan"


# %%
# Create dataframe to store dataframes of msg data
dfs = []
for key,value in get_paths().items():
    # Populate dataframe with messages from {name} parsed from JSON
    # print(key)
    # print(value)
    try:
        data = parse_single_json(key, name)
    # print(data)
        data["sender"] = value
        dfs.append(data)
    except:
        pass


# %%
# Combine data from dataframes dataframe into one dataframe
df_combined = pd.concat(dfs, sort=True)


# %%
# Add more calculated collumns
df_combined['timestamp_ms'] = pd.to_datetime(df_combined['timestamp_ms'], unit='ms') # set timestamp datatype
df_combined['date'] = df_combined['timestamp_ms'].apply(lambda x: (x + timedelta(hours=8)).date()) # calculate date from timestamp
df_combined['day_of_week'] = df_combined['timestamp_ms'].dt.day_name() # calculate day of week from timestamp
df_combined['character_count'] = df_combined['content'].str.len() # calculate character count
df_combined['word_count'] = df_combined['content'].apply(lambda s : len(str(s).split(' '))) # calculate wordcount based on spaces


# %%
# Sort data by day
df_combined['day_of_week'] = pd.Categorical(df_combined['day_of_week'], categories=
    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday'],
    ordered=True)


# %%
df_combined['content'] = df_combined['content'].apply(lambda x : str(x).replace(r'[^\w\s]+', ''))


# %%
# Drop all unneeded columns & reset index
df_combined = df_combined.reset_index(drop=False)
print(df_combined)
df_photos = df_combined[df_combined['photos'].isnull() == 0]
df_videos = df_combined[df_combined['videos'].isnull() == 0]
df_combined = df_combined.drop(['audio_files', 'call_duration', 'files','gifs','missed','reactions','share','sticker','users','videos','photos','type','index'], axis=1)


# %%
# Drop all content null rows
df_combined = df_combined[df_combined["content"].isnull() == 0]


# %%
df_photos = df_photos['photos'].apply(lambda x : len(x))
print(df_photos)


# %%
df_videos = df_videos['videos'].apply(lambda x : len(x))
print(df_videos)


# %%
print(df_combined.count())


# %%
df_wordcount_series = df_combined['content'].str.split(expand=True)
print(df_wordcount_series)


# %%
df_wordcount_series = df_wordcount_series.stack()
print(df_wordcount_series)


# %%
df_wordcount_series = df_wordcount_series.str.replace(r'[^\w\s]+', '')
df_wordcount_series = df_wordcount_series.str.lower()
print(df_wordcount_series)


# %%
df_wordcount_series = df_wordcount_series.value_counts()
print(df_wordcount_series)


# %%
df_wordcount_series['fuck']


# %%
top20words = df_wordcount_series.iloc[0:20]
print(top20words)


# %%
maxword = 'i'
maxword2 = 'i'
maxword3 = 'i'
for i in range(len(df_wordcount_series)):
    if len(df_wordcount_series.index[i]) > len(maxword):
        maxword3 = maxword2
        maxword2 = maxword

        maxword = df_wordcount_series.index[i]

print(maxword)
print(maxword2)
print(maxword3)



# %%
maxword = 'i'
maxword2 = 'i'
maxword3 = 'i'
for i in range(len(df_wordcount_series)):
    if len(df_wordcount_series.index[i]) > len(maxword):
        maxword3 = maxword2
        maxword2 = maxword

        maxword = df_wordcount_series.index[i]

print(maxword)
print(maxword2)
print(maxword3)



# %%
from datetime import datetime, timedelta

df_combined['timestamp_hkt'] = df_combined['timestamp_ms'].apply(lambda x: (x + timedelta(hours=8))) # set timestamp datatype

print(df_combined)


# %%
import cufflinks as cf
import plotly.offline
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# %%
df_combined['hour_of_day'] = df_combined['timestamp_hkt'].apply(lambda x : x.strftime('%H'))
df_combined.groupby(['hour_of_day'])['content'].count().iplot(dimensions=(1500,1000),colors=["DarkOrchid",],kind='bar',title="Texts on Hour",yTitle="Frequency",xTitle="Hour")


# %%
df_combined.groupby(['date'])['content'].count().iplot(dimensions=(1500,1000),colors=["MediumTurquoise",],kind='bar',title="Texts on Day",yTitle="Frequency",xTitle="Day")


# %%
df_combined.groupby(['day_of_week'])['content'].count().iplot(dimensions=(1500,1000),colors=["Aquamarine",],kind='bar',title="Messages on Day",yTitle="Frequency",xTitle="Day")


# %%
import cufflinks as cf
import plotly.offline
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# %%
top20words.iplot(dimensions=(1500,1000),subplots=True,colors=["red",],kind='bar',title="Most Commonly Used Words",yTitle="Frequency",xTitle="Word")


# %%
import plotly.graph_objects as go
fig = go.Figure(data=[go.Table(header=dict(values=['Metric', 'Value']),cells=dict( align='left',values=[["Total Number of Messages Sent","Number of Photos Sent","Number of Videos Sent","Total Number of Words Sent","Total Number of Characters Sent","Average Number of Messages Sent per Day","Average Word Count per Message", "Average Character Count per Message"], [df_combined.content.count(),df_photos.sum(),df_videos.sum(),df_combined.character_count.sum(),df_combined.word_count.sum(),df_combined.groupby(['date'])['content'].count().mean(),df_combined.word_count.mean(), df_combined.character_count.mean()]]))])
fig.show()


# %%
# df_combined.groupby(['sender'])['content'].count().reset_index(name='count') \
#                              .sort_values(['count'], ascending=False) \
#                              .head(5).iplot(colors=["LightSeaGreen",],kind='bar',title="Texts on Hour",yTitle="Frequency",xTitle="Hour")

df_combined.groupby(['sender'])['content'].count().reset_index(name='count').sort_values(['count'], ascending=False).set_index('sender').head(6).iplot(dimensions=(1500,1000),colors=["PaleGreen",],kind='bar',title="Messages Sent to Person",yTitle="Frequency",xTitle="Person")


# %%
a = "aaabaaa?"
a = a.replace(r'[^\w\s]+', '')
print(a)


# %%
words_list = df_combined['content'].values.tolist()
corpus = '\n'.join(words_list)
corpus = re.sub(r'[^\w\s]','',corpus)
print(corpus)


# %%
import markovify
text_model = markovify.NewlineText(corpus)


# %%
markov_array = []
for i in range(15):
    markov_array.append(text_model.make_short_sentence(320))


# %%
fig2 = go.Figure(data=[go.Table(header=dict(align='left',values=['No.', 'Generated Markov Chain']),cells=dict( align='left',values=[list(range(15)), markov_array]))])
fig2.show()

# %% [markdown]
# # LSTM TEXT GENERATION 

# %%
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


# %%
nlp_corpus = df_combined['content'].values.tolist()
nlp_corpus = ".".join(nlp_corpus).lower()
raw_text = nlp_corpus
import io
with io.open('aa.txt', "w", encoding="utf-8") as f:
    f.write(raw_text)


# %%
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)


# %%


# # define the LSTM model
# model = Sequential()
# model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
# model.add(Dropout(0.2))
# model.add(Dense(y.shape[1], activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam')
# # define the checkpoint
# filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]
# # fit the model
# model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)


# %%
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
# load the network weights
filename = "weights-improvement-01-2.7836.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print ("Seed:")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# generate characters
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print ("\nDone.")


# %%


