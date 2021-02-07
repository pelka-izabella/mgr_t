#%% 
from text_prepper import PrepareText
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from collections import Counter
from sklearn.manifold import TSNE

# %%
## Setup
in_dir = 'data'
out_dir = 'data'
#%%
# dataset - all Kraków restaurants scraped from Tripadvisor by Apify on 2020-10-03 16:10:41 
raw_dataset = 'prepped.csv'
df = pd.read_csv(os.path.join(in_dir, raw_dataset), index_col='Unnamed: 0', dtype = {'rating':np.int32, 'review':np.object})
df.head()

# %%
fig, ax = plt.subplots()
sns.countplot(df['rating'], color='forestgreen')

ax.set_xlabel('Ocena')
ax.set_ylabel('Liczba recenzji')
ax.set_title(r'Histogram ocen restauracji')

fig.tight_layout()
plt.show()
# %%
# counts of labels
def count_values_in_column(data,feature):
    total=data.loc[:,feature].value_counts(dropna=False)
    percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])

count_values_in_column(df,'rating')
# %%
# let's make 1-3 and 4-5 two classes
df['label'] = 0
df.loc[(df.rating == 4)|(df.rating == 5), "label"] = 1
# %%
fig, ax = plt.subplots()
sns.countplot(df['label'], color='navy')

ax.set_xlabel('Etykieta')
ax.set_ylabel('Liczba recenzji')
ax.set_title(r'Histogram etykiet')

fig.tight_layout()
plt.show()


#%%
# cleaning, tokenizing and lemmatizing
prep = PrepareText()
df = prep.clean_text(df, in_col='review', out_col='clean_text')
df = prep.lemmatize(df, in_col='clean_text', out_col='lem', listed=True)
df.head()

#%%
# wordcloud

positive_revs=False
negative_revs=True

if positive_revs==negative_revs==True:
    raise KeyError("Cannot filter out both positive aand negative")

if positive_revs:
    df = df[df['label'] == 1]
if negative_revs:
    df = df[df['label'] == 0]

lista_slow=[]
for rec in df.lem:
    for w in rec:
        lista_slow.append(w)

# list of unique words and frequencies
counter=Counter(lista_slow)
# deleting the word być as it doesn't add anything interesting to the picture
del counter["być"]

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='black',
                min_font_size = 10, colormap='YlGn', collocations=False).generate_from_frequencies(counter) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
# %%
