#%% 
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
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

# %%
# build term document matrix
V = len(df)
N = len(df)
sentences = df['review']

# create raw counts first
A = np.zeros((V, N))
print("V:", V, "N:", N)
j = 0
for sentence in sentences:
    for i in sentence:
        A[i,j] += 1
    j += 1
print("finished getting raw counts")

#%%
