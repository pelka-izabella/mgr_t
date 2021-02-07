#%% 
from text_prepper import PrepareText
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# %%
## Setup
in_dir = 'data'
out_dir = 'data'
raw_dataset = 'prepped.csv'

df = pd.read_csv(os.path.join(in_dir, raw_dataset), index_col='Unnamed: 0', dtype = {'rating':np.int32, 'review':np.object})

df['label'] = 0
df.loc[(df.rating == 4)|(df.rating == 5), "label"] = 1
labels = df['label']

prep = PrepareText()
df = prep.clean_text(df, in_col='review', out_col='clean_text')
df = prep.lemmatize(df, in_col='clean_text', out_col='lem', listed=False)
df.head()
# %%
# number of words in a review
df['word_count'] = df['review'].apply(lambda x: len(str(x).split(" ")))
df.head()
#%%
# average word length
def avg_word(sentence):
    words = sentence.split()
    return (sum(len(word) for word in words)/len(words))

df['avg_word_len'] = df['review'].apply(lambda x: avg_word(x))
# %%

tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',ngram_range=(1,1))
tfidf_trans = tfidf.fit_transform(df['lem']).toarray()
tfidf_trans

# %%
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
bow_trans = bow.fit_transform(df['lem']).toarray()
bow_trans
