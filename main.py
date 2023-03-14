
#%%

from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary

from gensim import models, corpora, similarities
import nltk
from nltk import FreqDist

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy

import pandas as pd
import utils


nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

#%%
experiment = None


data = pd.read_csv("data\\co_data.csv")

# arguments
CO_NAME = "Vahanalytics"
TOP_K = 5
NUM_TOPICS = 10
CHUNK = 5
TOKEN_SIZE = 5000 
experiment = "LDA"

if experiment == "TF-IDF":
    
    tfidf = TfidfVectorizer(utils.prep_process_tokenize).fit_transform(data["description"])
    query_index = utils.query_company(data, CO_NAME)
    similar_co_ind = utils.query_sim(tfidf, query_index, TOP_K)
    tfidf_result = utils.get_result(data, similar_co_ind)
 
elif experiment == "LDA":
    
    data_lda = utils.build_tokenized_col(data, stopwords)
    most_common_tokens = utils.most_tokens(data_lda, TOKEN_SIZE)
    training_set = utils.get_training_data(most_common_tokens, data_lda)
    
    dictionary = corpora.Dictionary(training_set)
    # dictionary = corpora.Dictionary(training_set["tokenized"])
    corpus = [dictionary.doc2bow(doc) for doc in training_set]
    lda = LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary, alpha=1e-2, eta=0.5e-2, chunksize=CHUNK)
    
    docmat_pad = utils.doc_matrix(lda, corpus, NUM_TOPICS)
    query_index = utils.query_company(data, CO_NAME)
    new_doc_mat = utils.get_dict2bow(data, lda, dictionary, query_index)
    query_answer = utils.matpad([new_doc_mat], NUM_TOPICS)[0]
    
    
    print("QUERY ANS", query_answer)
    print("QUERY.ANS.SHAPE", query_answer.shape)
    input()    
    
    most_sim_ids = utils.get_top_k_similar_docs(query_answer, docmat_pad, 2000, entropy, k=5)
    lda_result = utils.get_result(data, most_sim_ids)
 
# %%
