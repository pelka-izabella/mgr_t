#%% 
import pandas as pd
import nltk
import os
from stop_words import get_stop_words
import re
from nltk.tokenize import word_tokenize 
import morfeusz2
import itertools
from stempel import StempelStemmer # pystempel package
# import pl_stemmer as stem
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
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
# getting rid of stopwords, links, punctuation and so on

stop_words = get_stop_words('polish')
new_stopwords = ['i', 'a', 'w', 'z', 'ze']
stop_words.extend(new_stopwords)

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
# lemmatization
morf = morfeusz2.Morfeusz()

df['lem'] = ""
id=0


for rev in df['tokenized_text']: # iterating by reviews
    res = {}
    for word in rev: # iterating by words in a review
        analysis = morf.analyse(word) # different word forms
        trzon = []
        for interpretation in analysis: # analyzing each form
            trzon.append(interpretation[2][1].split(':')[0]) # getting rid of word endings
            trzon = list(set(trzon)) # only unique forms
            stem = dict(zip(itertools.repeat(word), trzon)) # matching one base form to the original word in review
            res.update(stem) 
    
    df['lem'][id] = list(res.values())

    id +=1

# note to self: stem = dict(zip(...)) leaves last form
df.head()

# %%
# stemming
# stemming to inny sposób doprowadzania słów do podstawowej, ujednoliconej wersji - pozbawia słowa formantów, zostawiając tylko 
# df['trzony'] = ''
# stemmer = StempelStemmer.default()
# # stemmer = StempelStemmer.polimorf()

# id=0

# for rev in df['tokenized_text']: # iterowanie po recenzjach
#     trzon=[]
#     for word in rev: # iterowanie po slowach
#         trzon.append(stemmer.stem(word))
            
#     df['trzony'][id] = trzon
#     id +=1

# df.head()


# #stem("zielony")
# # cos ten stemmer przestal dzialac

#%%
# wordcloud
lista_slow=[]
for rec in df.lem:
    for w in rec:
        lista_slow.append(w)

#slowa = " ".join([i for i in lista_slow])
# list of unique words and frequencies
counter=Counter(lista_slow)
# deleting the word być as it doesn't add anything interesting to the picture
del counter["być"]
#%%
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
 # write file
out_df = 'lemmatized.csv'
df.to_csv(os.path.join(out_dir, out_df))