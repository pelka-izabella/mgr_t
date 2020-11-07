#%% 
import pandas as pd
import nltk
import os
from stop_words import get_stop_words
import re
from nltk.tokenize import word_tokenize 
# %%
## Setup
in_dir = 'data'
out_dir = 'data'
#%%
# dataset - all Kraków restaurants scraped from Tripadvisor by Apify on 2020-10-03 16:10:41 
raw_dataset = 'prepped.csv'
df = pd.read_csv(os.path.join(in_dir, raw_dataset))
df.head()
# %%
# getting rid of stopwords (także, ja, bardzo etc.), links, punctuation and so on

stop_words = get_stop_words('polish')

def remove_stopwords(text):
    text = ' '.join([i for i in text.split(' ') if i not in stop_words])
    return text

def clean_text(text):
    '''This function removes URL, punctuation marks and digits, then converts the text into lowercase and applies remove_stopwords'''

    text = re.sub('https?://[A-Za-z0-9./]*','', text) # Remove https..(URL)
    # text = re.sub('RT @[\w]*:','', text) # Removed RT 
    # text = re.sub('@[A-Za-z0-9_]+', '', text) # Removed @mention
    # text = re.sub('#', '', text) # hastag into text
    text = re.sub('&amp; ','',text) # Removed &(and) 
    text = re.sub('[0-9]*','', text) # Removed digits
    text = re.sub('[^\w+]',' ',text) # remove non words
    text = text.strip().lower()
    text = remove_stopwords(text)
    return text
# %%
test = df['review'].to_list()

# for display purposes
# for i in test:
#     print(i)
#     print("-"*10)
#     print("clean_text \n")
#     print(clean_text(i))
#     print("\n")
#     print("*"*10)


df['clean_text'] = df['review'].apply(clean_text)

# %%

# those 3 lines below tokenize based on whole sentences, which doesn't make sense because I removed punctuation marks, leaving here cause could be useful
# nltk.download('punkt') 
# tokenizer = nltk.data.load('tokenizers/punkt/polish.pickle')
# df['tokenized'] = df.apply(lambda row: tokenizer.tokenize(row['clean_text']), axis=1)

# this tokenizer splits the review by words, so each word becomes a seperate token
df['tokenized_text'] = [word_tokenize(i) for i in df['clean_text'].to_list()]
df.head()

 # %%
out_df = 'tokenized.csv'
df.to_csv(os.path.join(out_dir, out_df))

# %%
