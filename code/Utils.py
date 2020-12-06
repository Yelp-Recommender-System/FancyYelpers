#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import re, string, unicodedata
import nltk
import contractions
# import inflect

# import lime
# from lime import lime_text
# from lime.lime_text import LimeTextExplainer

import re

from wordcloud import WordCloud, STOPWORDS

import nltk
# import contractions
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import SnowballStemmer
from nltk.stem import LancasterStemmer, WordNetLemmatizer


from sklearn import metrics
from sklearn.model_selection import train_test_split
# from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

import scipy

from gensim import corpora
from gensim import corpora
from gensim.similarities.docsim import Similarity
from gensim import corpora, models, similarities

import pickle
import time
#%%
### Text Prepocessing
def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
            
    return new_words

def remove_special(words):
    """Remove special signs like &*"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[-,$()#+&*]', '', word)
        if new_word != '':
            new_words.append(new_word)
            
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
            
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""  
    stopwords = nltk.corpus.stopwords.words('english')
    myStopWords = []
    stopwords.extend(myStopWords)
    new_words = []
    for word in words:
        if word not in stopwords:
            new_words.append(word)
            
    return new_words

def to_lowercase(words):
    """Convert words to lowercase"""
    new_words=[]
    for word in words:
        new_words.append(word.lower())
        
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stemmer = SnowballStemmer('english')
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
        
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
        
    return lemmas

def normalize_lemmatize(words):
    words = remove_special(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    words = stem_words(words)
    words = lemmatize_verbs(words)
    return words

def get_processed(sample):
    processed = pd.DataFrame(data=[],columns = ['business_id', 'text'])
    new_texts = []

    for i in range(0, len(sample)):
        business_id = sample['business_id'].iloc[i]
        words = nltk.word_tokenize(sample['text'].iloc[i])
        text = ' '.join(normalize_lemmatize(words))
        dfnew = pd.DataFrame([[business_id, text]], columns=['business_id', 'text'])
        new_texts.append(text)
        processed = processed.append(dfnew,ignore_index = True)
    return processed
#%%
## Similarity matrix

def get_tfidf_matrix(processed):
    '''
    get the Tf-Idf matrix of processed texts for business reviews
    
    '''
    TV = TfidfVectorizer(stop_words = "english")
    processed["text"] = processed["text"].fillna('')
    tfidf_matrix = TV.fit_transform((processed["text"]))
    
    return tfidf_matrix

def get_cos_sim_matrix(tfidf_matrix, n):
    '''
    use truncated SVD to reduce dimensions to n 
    @n: the dimensions to keep
    '''
    SVD = TruncatedSVD(n_components = n , random_state = 42) # 42 is the ultimate answer to everything
    tfidf_truncated = SVD.fit_transform(tfidf_matrix)
    cosine_sim = cosine_similarity(tfidf_truncated, tfidf_truncated)
    
    return cosine_sim

def get_euclidean_sim(business, n_components):
    
    SVD = TruncatedSVD(n_components = n_components , random_state = 42) # 42 is the ultimate answer to everything
    bus_truncated = SVD.fit_transform(business)
    
    eucl_dist = euclidean_distances(bus_truncated)
    eucl_sim = 1/np.exp(eucl_dist)
    return eucl_sim
    
def get_buscosine_sim(business,n_components):
    SVD = TruncatedSVD(n_components = n_components , random_state = 42) # 42 is the ultimate answer to everything
    bus_truncated = SVD.fit_transform(business)
    
    cosine_sim = cosine_similarity(bus_truncated, bus_truncated)
    return cosine_sim

def get_mix_sim_matrix(cosine_sim, bus_cos_sim, lmbda = 0.5, ):
    mixed_sim = np.add(cosine_sim*lmbda, bus_cos_sim*(1-lmbda)) # assume equally weighted

    return mixed_sim
def get_mix_sim_df(df_tfidf_sim, df_bus_sim, lmbda = 0.5):
    df_mixed_sim = lmbda * df_tfidf_sim + (1-lmbda) * df_bus_sim.loc[df_tfidf_sim.index, df_tfidf_sim.columns]
    return df_mixed_sim

#%%
## Get recommendations and prediction
def get_recommendation_cos(reviews, business_id, user_id, df_sim, k):
    '''get the business_id_array that shows top_k greatest similarity to the specific business_id'''
    user_bids = reviews[reviews['user_id']==user_id]['business_id'].values
    df_user = df_sim.loc[df_sim.index.isin(user_bids), df_sim.columns == business_id]
    df_user_topk = df_user.sort_values(df_user.columns[0], ascending = False).iloc[:k]
    
    return np.array(df_user_topk.index.values)


def predict_rating(reviews, user_id, business_ids):
    '''predict the avg of the user's rating on business in business_ids'''
    scores = reviews.loc[(reviews.user_id == user_id) & (reviews.business_id.isin(business_ids))]["stars"].values
    
    return np.mean(scores)


def get_results_cos(reviews, reviews_test, business_id, user_id, df_sim, k):
    '''
    prediction on the business_id：avg the ratings on top_k business that shows similarity to the business_id
    actual on the business_id: the true rating 
    '''
    actual = reviews_test.loc[(reviews_test.user_id==user_id) & (reviews_test.business_id==business_id)]['stars'].values[0]
    business_ids = get_recommendation_cos(reviews, business_id, user_id, df_sim, k)
    prediction = predict_rating(reviews, user_id, business_ids)
    
    return actual, prediction

def get_review_processed(processed, reviews):
    reviews_processed = reviews.loc[reviews.business_id.isin(processed.business_id)]\
                                                       .reset_index()\
                                                       .drop(columns=['index'])
    return reviews_processed

def CB_predict(reviews, reviews_test, df_sim, k = 5):
    '''
    based on test_df 
    get a dataframe with each user on each business's true ratings and prediction ratings
    @df_sim, n*n DataFrame for business similarities
    @k: int, top k similar businesses
    '''
    user_id_sample = reviews_test['user_id'].values
    busi_id_sample = reviews_test['business_id'].values
    
    actual = []
    predictions = []
    
    for i in range(len(reviews_test)):
        try:
            act, pred = get_results_cos(reviews, reviews_test, busi_id_sample[i], user_id_sample[i], df_sim, k)
            actual.append(act)
            predictions.append(pred)
            
        except:
            actual.append(np.nan)
            predictions.append(np.nan)
            
    return pd.DataFrame({"user_id": user_id_sample,
                         "business_id": busi_id_sample,
                         "true_ratings": actual,
                         "prediction_ratings":  predictions                        
                        })

#%%
## LSI model
def get_lsi(processed, reviews, user_id, n_topics):
    '''
    get the lsi model for user_id
    '''
    user_bids = reviews[reviews['user_id']==user_id]['business_id'].values
    processed_user = processed.loc[processed.business_id.isin(user_bids)]
    documents = list(processed_user['text'].values)
    texts = [[word for word in document.split(' ')] for document in documents]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts] 

    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=n_topics)
    
    return lsi, dictionary, corpus

def get_recommendation_lsi(processed, reviews, df_lsi, business_id, user_id, k, n_topics):
    lsi = df_lsi[(df_lsi["n_topic"] == n_topics) & (df_lsi["user_id"] == user_id)]["lsi"][0]
    dictionary = df_lsi[(df_lsi["n_topic"] == n_topics) & (df_lsi["user_id"] == user_id)]["dictionary"][0]
    user_bids = reviews[reviews['user_id']==user_id]['business_id'].values
    processed_user = processed.loc[processed.business_id.isin(user_bids)]
    documents = list(processed_user['text'].values)
    texts = [[word for word in document.split(' ')] for document in documents]
    corpus = [dictionary.doc2bow(text) for text in texts] 
    
    doc = processed['text'].loc[processed.business_id==business_id].values[0]
    vec_bow = dictionary.doc2bow(doc.lower().split())
    vec_lsi = lsi[vec_bow]
    index = similarities.MatrixSimilarity(lsi[corpus])
    sims = list(index[vec_lsi])
    results = list(zip(user_bids, sims))
    results_ordered = np.array(sorted(results, key=lambda x: x[1], reverse=True))
    results_topk = results_ordered[:k]
    
    return results_topk[:,0]

def get_results_lsi(processed,reviews,reviews_test, df_lsi ,business_id,user_id,k,n_topics):
    actual = reviews_test.loc[(reviews_test.user_id==user_id) & (reviews_test.business_id==business_id)]['stars'].values[0]
    business_ids = get_recommendation_lsi(processed,reviews,df_lsi ,business_id,user_id,k,n_topics)
    prediction = predict_rating(reviews, user_id, business_ids)
    return actual, prediction

def get_recommendation_lsi(processed,reviews,business_id,user_id,k,n_topics):
    user_bids = reviews[reviews['user_id']==user_id]['business_id'].values
    processed_user = processed.loc[processed.business_id.isin(user_bids)]
    documents = list(processed_user['text'].values)
    texts = [[word for word in document.split(' ')] for document in documents]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts] 

    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=n_topics)
    doc = processed['text'].loc[processed.business_id==business_id].values[0]
    vec_bow = dictionary.doc2bow(doc.lower().split())
    vec_lsi = lsi[vec_bow]
    index = similarities.MatrixSimilarity(lsi[corpus])
    sims = list(index[vec_lsi])
    results = list(zip(user_bids, sims))
    results_ordered = np.array(sorted(results, key=lambda x: x[1], reverse=True))
    results_topk = results_ordered[:k]
    
    return results_topk[:,0]

def get_results_lsi(processed,reviews,reviews_test ,business_id,user_id,k,n_topics):
    actual = reviews_test.loc[(reviews_test.user_id==user_id) & (reviews_test.business_id==business_id)]['stars'].values[0]
    business_ids = get_recommendation_lsi(processed,reviews,business_id,user_id,k,n_topics)
    prediction = predict_rating(reviews, user_id, business_ids)
    return actual, prediction

def CB_LSI_predict(df_texts_train, reviews, reviews_test, k = 5, n_topics = 100):
    uid_sample = reviews_test['user_id'].values
    bid_sample = reviews_test['business_id'].values
    actual_lsi = []
    predictions_lsi = []
    for i in range(len(reviews_test)):
        try:
            act, pred = get_results_lsi(df_texts_train, reviews, reviews_test, bid_sample[i],uid_sample[i], k, n_topics)
            predictions_lsi.append(pred)
            actual_lsi.append(act)

        except:
            predictions_lsi.append(np.nan)
            actual_lsi.append(np.nan)
            
    return pd.DataFrame({"user_id": uid_sample,
                         "business_id": bid_sample,
                         "ratings": actual_lsi,
                         "pred_lsi":  predictions_lsi})


#%% Get Full predictions
def get_recommendation_cos_full(reviews, user_id, df_sim, k, busi_id_lst):
    '''get the business_id_array that shows top_k greatest similarity to the specific business_id'''
    
    df_user_rating = reviews[reviews.user_id == user_id]
#     df_sim = df_sim.loc[busi_id_lst, busi_id_lst]
    user_bids = df_user_rating['business_id'].values
    df_user = df_sim.loc[df_sim.index.isin(user_bids)]
    df_user_rank = df_user.rank(ascending = False, axis = 0)
    df_user_rank[df_user_rank <= k] = 1
    df_user_rank[df_user_rank > k] = 0
    df_user_rank = df_user_rank/ np.min([k, len(user_bids)])
  
    user_rating_matrix = np.array(df_user_rating[["business_id", "stars"]].set_index(["business_id"]).loc[df_user_rank.index.values])
    pred = user_rating_matrix.T @ np.array(df_user_rank)
    return pred

def CB_sim_fit_full_matrix(train_valid_df, df_sim, k, user_id_lst, busi_id_lst):
    rating_pred_matrix = np.zeros((len(user_id_lst), len(busi_id_lst)))
    for i,user_id in enumerate(user_id_lst):
        rating_pred_matrix[i,] = get_recommendation_cos_full(train_valid_df, user_id, df_sim, k, busi_id_lst)
    return(rating_pred_matrix)

def get_mse(pred, actual):
    # Ignore zero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

