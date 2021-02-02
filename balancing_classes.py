#%% 
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
# %%
## Setup
in_dir = 'data'
out_dir = 'data'
#%%
# dataset 
raw_dataset = 'prepped.csv'
df = pd.read_csv(os.path.join(in_dir, raw_dataset), index_col='Unnamed: 0', dtype = {'rating':np.int32, 'review':np.object})
df['label'] = 0
df.loc[(df.rating == 4)|(df.rating == 5), "label"] = 1
# %%
df.drop(columns='rating', inplace=True)
df.head()
# %%
# upsampling
print("Raw count of classes \n", df.label.value_counts())
# Separate majority and minority classes
df_majority = df[df.label==1]
df_minority = df[df.label==0]

# %%
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=8709,    # to match majority class
                                 random_state=42) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
print("Count of classes in upsampling \n",df_upsampled.label.value_counts())
# %%
# SMOTE
# Oversample with SMOTE and random undersample for imbalanced dataset

# define dataset
X = df['review']
y = df['label']
#%%
# summarize class distribution
counter = Counter(y)
print(counter)
#%%
# define pipeline
over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
#%%
# transform the dataset
X, y = pipeline.fit_resample(X, y)
# %%
# summarize the new class distribution
counter = Counter(y)
print(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = np.where(y == label)[0]
	plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
plt.legend()
plt.show()
# %%
