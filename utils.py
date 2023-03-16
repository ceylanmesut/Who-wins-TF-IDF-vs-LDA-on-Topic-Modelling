

import re
import nltk

import numpy as np
import pandas as pd
# from  collections import Counter

from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')


def prep_process_tokenize(text, stopwords):
    
    """Text cleaner.
    Args:
        text: dataframe, the company data
        stopwords: list, english stopwords
    Returns:
        text: lower-cased, tokenized and processed text 
    """
    
    # cleaning the URLs, email and any punctuation existing
    text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)
    text = re.sub("[^a-zA-Z ]", "", text)
    text = text.lower() # lower casing the text
    text = nltk.word_tokenize(text) # tokenizing
    
    # removing stopwords
    text = [word for word in text if word not in stopwords]
    
    return text

def matpad(docmat, num_topics):
    
    """Document matrix padder.
    Args:
        docmat: array, document matrix
        num_topics: int, number of topics
    Returns:
        stub_mat: array, padded matrix
    """
    
    stub_mat = []
    for doc in docmat:
        present = set(map(lambda x: x[0], doc))
        if present == num_topics:
            stub_mat.append(doc)
            continue
  
        topic_sum = sum(map(lambda x: x[1], doc))
        reminders = (1 - topic_sum) / (num_topics - len(present)) # TODO: zero encountered in scalar divide
        d_doc = dict(doc)
        stub_mat.append([d_doc[i] if i in present else reminders for i in range(num_topics)])
        
    return np.asarray(stub_mat)

def pre_process(text, stopwords):
    
    """Text preprocessor.
    Args:
        text: dataframe, the company data
        stopwords: list, english stopwords
    Returns:
        text: lower-cased, tokenized and processed text 
    """
    
    return " ".join(prep_process_tokenize(text,stopwords))

def build_tokenized_col(data, stopwords):
    
    """Builds token column.
    Args:
        data: dataframe, the company data
        stopwords: list, english stopwords
    Returns:
        data: dataframe with token column
    """        
    
    data["tokenized"] = np.nan
    for index in (range(len(data["description"]))):
        data["tokenized"][index] = pre_process(data["description"][index], stopwords)

    return(data)

def get_training_data(data):
    
    """Builds training data.
    Args:
        data: dataframe, the company data
    Returns:
        data: pandas series, tokens
    """    

    for index in range(len(data["tokenized"])):

        # splitting the tokens in each row of df
        tokens = data["tokenized"][index].split()
        
        # replacing column values with the new column values
        data["tokenized"][index] = list(tokens)
        data_to_train = data["tokenized"]
    
    return(data_to_train)

def get_corpus(data, dictionary):
    
    """Builds corpus for LDA model.
    Args:
        data: dataframe, the company data
        dictionary: gensim dict obj, dictionary of training set 
        query_idx: int, query caompany index
    Returns:
        corpus: tuple, word_id and word frequency
    """      
    
    corpus = [dictionary.doc2bow(doc) for doc in data]    
    return corpus

def get_dict2bow(data, lda, dictionary, query_idx):
    
    """Builds bag of words from gensim dictionary.
    Args:
        data: dataframe, the company data
        lda: gensim lda obj, trained lda model 
        dictionary: gensim dict obj, dictionary of training set 
        query_idx: int, query caompany index
    Returns:
        topic_dist: tuple, topic distribution: topic_id, topic probability 
    """    
    # turning dictionary to bag of words of query company
    query_bow = dictionary.doc2bow(data.loc[query_idx]["tokenized"])
    # obtaining topic distribution of the query company
    topic_dist = lda.get_document_topics(bow=query_bow)
    
    return(topic_dist)
    
def doc_matrix(trained_model, corpus, num_topics):
    
    """Builds document matrix.
    Args:
        trained_model: gensim lda obj, trained LDA model
        corpus: array, word_id and word frequency
        num_topics: int, number of topics
    Returns:
        docmat_pad: array, padded document matrix, shape n x num.of.topics
    """       
    
    docmat = ([x for x in trained_model.get_document_topics(trained_model[corpus])])
    docmat_pad = matpad(docmat, num_topics)
    
    return(docmat_pad, docmat)

def jensen_shannon(query, matrix, num_documents, entropy):
    
    """Calculates Jensen-Shannon Divergence.
    Args:
        query: array, query answer from LDA
        matrix: array, word_id and word frequency
        num_documents: int, number of documents
        entropy: scipy.stats obj, entropy 
    Returns:
        js_divergence: computed js_div scores
    """       
    
    p = np.matrix([query for i in range(num_documents)]).T
    q = matrix.T 
    m = 0.5 * (p + q)
    
    return np.sqrt(0.5*(entropy(p, m) + entropy(q, m)))

def get_top_k_similar_docs(query, matrix, num_documents, entropy, k=10):
    
    """Gets Top k similar documents.
    Args:
        query: array, query answer from LDA
        matrix: array, word_id and word frequency
        num_documents: int, number of documents
        entropy: scipy.stats obj, entropy
        k: int,  top k similarity threshold
    Returns:
        similar_documents: top k similar documents
    """       
    
    sims = jensen_shannon(query, matrix, num_documents, entropy)
    
    return(sims.argsort()[:k], sims)


def query_idx(data, co_name):
    
    """Fetches the index of the query company name.
    Args:
        data: dataframe, the company data
        co_name: str, name of the query company, e.g:"Vahanalytics"
    Returns:
        index: int, query company index
    """
    
    co_index = data.index[data['name'] == co_name].tolist()[0]    
    return co_index

def query_sim(tf_idf, query_index, top_k):
    
    """Cosine similarity calculator.
    Args:
        tf_idf: tf-idf object, fitted and transformed tf-idf object
        query_index: int, query company index
        top_k: int, top k similarity threshold
    Returns:
        related_docs_ind: array, indicies of the top similar documents 
        cos_sim: array, cosine similarities
    """    
    
    cos_sim = cosine_similarity(tf_idf[query_index: query_index + 1], tf_idf).flatten()
    related_docs_ind = cos_sim.argsort()[:-top_k - 1:-1]   

    return(related_docs_ind, cos_sim)

def get_result(experiment, data, sim_ind, top_k):
    
    """Builds result dataframe.
    Args:
        experiment: str, name of the experiment
        data: dataframe, the company data
        sim_ind: array, indicies of the top similar documents 
        top_k: int, top k similarirty threshold
    Returns:
        result: dataframe, contains top k similar company descriptions and names
    """    
    
    result = pd.DataFrame(columns=["name", "description"])

    for ind in sim_ind:
        answer = data[["name", "description"]][data.index==ind]
        result = pd.concat([result, answer])
    
    if experiment == "TF-IDF": 
        result["Cosine Similarities"] = sorted(sim_ind, reverse=True)[: top_k]

    return result 
    
     