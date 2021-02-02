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
# %%
X_train, X_test, y_train, y_test = train_test_split(X=bow_trans, y=labels, test_size=0.2, random_state=0)
# %%
rf = RandomForestClassifier(n_estimators=200, random_state=0)
rf.fit(X_train, y_train)
# %%
y_pred = rf.predict(X_test)

# %%

def plot_confusion_matrix(y_test, y_pred):
    cf_matrix = confusion_matrix(y_test, y_pred)
    group_names = ["True Neg","False Pos","False Neg","True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                        cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.show()

    print("Classification report \n" classification_report(y_test,y_pred))
    print("Accuracy: ", round(accuracy_score(y_test, y_pred)*100, 4), "%")
# %%
plot_confusion_matrix(y_test, y_pred)
# %%
