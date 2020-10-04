#%% 
import pandas as pd
import os
from langdetect import detect
# %%
## Setup
in_dir = 'data'
out_dir = 'data'
#%%
# dataset - all Kraków restaurants scraped from Tripadvisor by Apify on 2020-10-03 16:10:41 
raw_dataset = 'all_krakow_raw.csv'
df = pd.read_csv(os.path.join(in_dir, raw_dataset))
df.head()

# we need places with reviews
df = df[df['numberOfReviews']>=1]
# %%
# filtering out the restaurants that are vegetarian friendly or have vegan options

df = df[df['category'] == "restaurant"]
veg = ['Vegan Options', 'Vegetarian Friendly']
veg_cols = ['cuisine/0','cuisine/1', 'cuisine/2','cuisine/3','cuisine/4','cuisine/5','cuisine/6','cuisine/7','cuisine/8','cuisine/10']

# not the nicest way to code it
df_vv =  df[df[veg_cols].apply(lambda r: r.str.contains('Vegan Options', case=False).any(), axis=1)] 

df_v =  df[df[veg_cols].apply(lambda r: r.str.contains('Vegetarian Friendly', case=False).any(), axis=1)] 

cols_to_join = df_v.columns.to_list()
df_veg= df_v.merge(df_vv, how='outer', on=cols_to_join).drop_duplicates()

# %%
# let's get rid of unnecessary columns, leaving only review data
rev =  ['name'] + [col for col in df_veg.columns if 'reviews' in col]

df_rev = df_veg[rev]

# %%
# epic quest to bring the data into the right format

#cols_to_unpiv = [col for col in df_rev.columns if 'text' in col] 
df_text = df_rev.melt(id_vars=['name']) #  value_vars = cols_to_unpiv, var_name='text'

# extracting review id per name and type of data (title, text etc)
df_text['id'] = df_text['variable'].str.split('/', expand=True)[1]
df_text['type'] = df_text['variable'].str.split('/', expand=True)[2]
df_text.drop(columns=['variable'], inplace=True)
df_text
# %%
# some cleaning up
df_text.set_index(['name', 'id'], inplace=True)

cols_to_keep=['rating', 'text', 'title'] #,'userLocation'
df_text_clean = df_text[df_text['type'].isin(cols_to_keep)]
df_text_clean
# %%
# finally getting to the right type
df_text_clean = df_text_clean.pivot_table(values='value', index=df_text_clean.index, columns='type', aggfunc='first')
# %%
# adding the title of the review into the text
df_text_clean['review'] = df_text_clean['title'] + " " + df_text_clean['text']
df_text_clean.drop(columns=['title', 'text'], inplace=True)

# %%
# we'd like to see the reviews in Polish - since there is no review language indicator, let's try to detect it, otherwise, I'll use user location being in Poland as a proxy for now
df_text_clean['lang'] = df_text_clean['review'].apply(detect)
df_pl = df_text_clean[df_text_clean['lang'] == 'pl']
df_pl.drop(columns='lang', inplace=True)
# and done