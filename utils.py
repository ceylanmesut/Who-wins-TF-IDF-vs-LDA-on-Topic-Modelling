

import re
import nltk

import numpy as np
import pandas as pd
from  collections import Counter

from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')


def prep_process_tokenize(text):
    
    """Text cleaner.
    Args:
        text: dataframe, the company data
        stopwords: list, english stopwords
    Returns:
        text: lower-cased, tokenized and processed text 
    """
    
    # cleaning the URLs, email and any punctuation existing
    text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", str(text))
    text = re.sub("[^a-zA-Z ]", "", str(text))
    text = re.sub("[0-9]", "", str(text))
    text = text.lower() # lower casing the text
    text = nltk.word_tokenize(text) # tokenizing
    
    stopwords = nltk.corpus.stopwords.words('english')
    ext_stopwords = extend_stopwords(stopwords)
    
    
    # removing stopwords
    text = [word for word in text if word not in ext_stopwords]
    
    lemmatizer = nltk.stem.WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]    
    
    stemmer = nltk.stem.PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    
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
        
        eps = 0.0000001
        topic_sum = sum(map(lambda x: x[1], doc))
        reminders = (1 - topic_sum) / ((num_topics - len(present)) + eps)
        d_doc = dict(doc)
        stub_mat.append([d_doc[i] if i in present else reminders for i in range(num_topics)])
        
    return np.asarray(stub_mat)

def pre_process(text):
    
    """Text preprocessor.
    Args:
        text: dataframe, the company data
    Returns:
        text: lower-cased, tokenized and processed text 
    """
    
    return " ".join(prep_process_tokenize(text))

def build_tokenized_col(data):
    
    """Builds token column.
    Args:
        data: dataframe, the company data
    Returns:
        data: dataframe with token column
    """        
    
    data["tokenized"] = np.nan
    for index in (range(len(data["description"]))):
        data["tokenized"][index] = pre_process(data["description"][index])

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

def get_result_tfidf(data, sim_ind, cosine_sims, top_k):
    
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
    
    result["Cosine Similarities"] = sorted(cosine_sims, reverse=True)[: top_k]
    result.to_excel("tf_idf_results.xlsx")

    return result


def save_tf_idf_scores(tfidf, vectorizer, query_index):
    
    tfidf_output = tfidf[query_index]
    df_tfidf = pd.DataFrame(tfidf_output.T.todense(), index=vectorizer.get_feature_names_out(), columns=['tf-idf'])
    analysis = df_tfidf.sort_values(by=["tf-idf"], ascending=False)
    
    scores = analysis.head(20)
    scores.to_excel("tf_idf_scores_%s.xlsx" % query_index)


def get_result_lda(data, sim_ind):
    
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
    
    result.to_excel("LDA_results.xlsx")
        
    return result 
    
    
def extend_stopwords(stopwords):
    
    # data = pd.read_csv("data\\co_data.csv")
    
    # corpus = []
    # for i in range(len(data)):
    #     for word in data.description[i].split():
    #         corpus.append(word)
    
    # word_count = Counter(corpus)
    
    # most_common_words = []
    # most_common_count = []

    # for i in range(len(word_count.most_common(30))):
    #     most_common_words.append(word_count.most_common(30)[i][0])
    #     most_common_count.append(word_count.most_common(30)[i][1])
        

    # stopwords.extend(most_common_words)
     
    most_common_words  =['services', 'company', 'construction','management','logistics','service',
    'solutions','provides','products','customers', 'customers','platform','industry','business','also','offers','software','project',
    'technology','delivery','equipment','companies','building','design','projects','engineering','quality','include',
    'provide','based','new', 'us', 'clients', 'solution', 'global', 'international', 'well', 'systems', 'group', 'founded',
    'users', 'market', 'product', 'provider', 'one']
    
    stopwords.extend(most_common_words)
    
    return(stopwords)
    
    