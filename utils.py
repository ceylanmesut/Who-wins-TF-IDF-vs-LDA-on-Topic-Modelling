

import re
import nltk

import numpy as np
import pandas as pd
from  collections import Counter

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')


def prep_process_tokenize(text, stopwords):
    
    #websites, email and any punctuation cleaning
    text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)
    text = re.sub("[^a-zA-Z ]", "", text)
    text = text.lower() # lower case the text
    text = nltk.word_tokenize(text) # tokenizing
    
    #removing stopwords
    text = [word for word in text if word not in stopwords]
    
    return text


def matpad(docmat, num_topics):
  stub_mat = []
  for doc in docmat:
    present = set(map(lambda x: x[0], doc))
    if present == num_topics:
      stub_mat.append(doc)
      continue
    topic_sum = sum(map(lambda x: x[1], doc))
    reminders = (1 - topic_sum) / (num_topics - len(present))
    d_doc = dict(doc)
    stub_mat.append([d_doc[i] if i in present else reminders for i in range(num_topics)])
    
  return np.asarray(stub_mat)

def pre_process(text, stopwords):
    return " ".join(prep_process_tokenize(text,stopwords))

def build_tokenized_col(data, stopwords):
    
    data["tokenized"] = np.nan
    for index in (range(len(data["description"]))):
        data["tokenized"][index] = pre_process(data["description"][index], stopwords)

    return(data)

def most_tokens(data, top_n=5000):
    
    tokens_list = []

    for index in range(len(data["tokenized"])):
        tokens = data["tokenized"][index].split()
        
        for token in tokens:
            tokens_list.append(token)
            
    counted_tokens = Counter(tokens_list)
    most_common_tokens = counted_tokens.most_common(top_n)

    most_common_tokens = []
    for index in range(len(most_common_tokens)):
        most_common_tokens.append(most_common_tokens[index][0]) 

    return(most_common_tokens)

def get_training_data(common_tokens, data):
    # Filtering uncommon words from column values
    most_common_5000_tokens_set  = set(common_tokens)

    for index in range(len(data["tokenized"])):

        # splitting the tokens in each row of df
        tokens = data["tokenized"][index].split()
        #tokens = set(tokens)
        
        #filtered_tokens = most_common_5000_tokens_set & tokens

        # replacing column values with the new column values
        data["tokenized"][index] = list(tokens)#(filtered_tokens)
        
        data_to_train = data["tokenized"] #.astype("string")                  
    
    return(data_to_train)


def get_dict2bow(data, lda, dictionary, query_idx):
    
    print("QUERY_IDX", type(query_idx))
    print("DATA CORR", data.loc[query_idx]["tokenized"])
    new_bow_Vahanalytics = dictionary.doc2bow(data.loc[query_idx]["tokenized"])
    new_doc_Vahanalytics = lda.get_document_topics(bow=new_bow_Vahanalytics)    
    
    return(new_doc_Vahanalytics)
    
def doc_matrix(trained_model, corpus, num_topics):
    
    docmat = ([x for x in trained_model.get_document_topics(trained_model[corpus])])
    docmat_pad = matpad(docmat, num_topics)
    
    return(docmat_pad)

def jensen_shannon(query, matrix, num_documents, entropy):
    
    p = np.matrix([query for i in range(num_documents)]).T
    q = matrix.T 
    m = 0.5*(p + q)
    
    return np.sqrt(0.5*(entropy(p, m) + entropy(q, m)))

def get_top_k_similar_docs(query, matrix, num_documents, entropy, k=10):
    
    sims = jensen_shannon(query, matrix, num_documents, entropy)
    
    return sims.argsort()[:k]


def query_company(data, co_name):
    
    co_index = data.index[data['name'] == co_name].tolist()[0]    
    return co_index

def query_sim(tf_idf, query_index, top_k):
    
    cosine_similarities = cosine_similarity(tf_idf[query_index: query_index + 1], tf_idf).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-top_k - 1:-1]   

    return(related_docs_indices)

def get_result(data, sim_ind):
    
    result_df = data[data.index.isin(sim_ind)]  

    return result_df 
    
     