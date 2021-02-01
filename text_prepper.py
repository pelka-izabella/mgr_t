import pandas as pd
import nltk
import os
from stop_words import get_stop_words
import re
from nltk.tokenize import word_tokenize 
import morfeusz2
import itertools
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter


class PrepareText:

    def __init__(self):
        self
    
    @staticmethod
    def clean_text(text):
        '''This function removes URL, punctuation marks and digits, then converts the text into lowercase and applies remove_stopwords'''
        
        text = re.sub('https?://[A-Za-z0-9./]*','', text) # Remove https..(URL)
        text = re.sub('&amp; ','',text) # Removed ampersand 
        text = re.sub('[0-9]*','', text) # Removed digits
        text = re.sub('[^\w+]',' ',text) # remove non words
        text = text.strip().lower()
        stop_words = get_stop_words('polish')
        new_stopwords = ['i', 'a', 'w', 'z', 'ze']
        stop_words.extend(new_stopwords)
        text = ' '.join([i for i in text.split(' ') if i not in stop_words])
        
        return text


    @staticmethod
    def tokenize(df, in_col='clean_text', out_col='tokenized', drop_incol=False):
        """Tokenizing using word tokenizer"""
        df[out_col] = [word_tokenize(i) for i in df[in_col].to_list()]
        if drop_incol:
            df = df.drop(in_col)
        return df

    @staticmethod
    def lemmatize(df, in_col='tokenized', out_col='lem', drop_incol=False):
        """This function turns Polish words into their most basic form. The POS of the output is not being defined"""

        morf = morfeusz2.Morfeusz()
        df[out_col] = ""
        id=0

        for rev in df[in_col]: # iterating by reviews
            res = {}
            for word in rev: # iterating by words in a review
                analysis = morf.analyse(word) # different word forms
                trzon = []
                for interpretation in analysis: # analyzing each form
                    trzon.append(interpretation[2][1].split(':')[0]) # getting rid of word endings
                    trzon = list(set(trzon)) # only unique forms
                    # note to self: stem = dict(zip(...)) leaves last form
                    stem = dict(zip(itertools.repeat(word), trzon)) # matching one base form to the original word in review
                    res.update(stem) 
            
            df[out_col][id] = list(res.values())
            id +=1
        
        if drop_incol:
            df = df.drop(in_col)

        return df

