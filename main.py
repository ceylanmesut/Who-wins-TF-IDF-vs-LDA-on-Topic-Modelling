
#%%


from gensim.models import LdaModel
from gensim import models, corpora
import nltk


from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import entropy

import pandas as pd
import utils

# nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

data = pd.read_csv("data\\co_data.csv")

# arguments
CO_NAME = "Vahanalytics"
TOP_K = 5
NUM_TOPICS = 4
CHUNK = 5
TOKEN_SIZE = 15000 
experiment = "LDA"

if experiment == "TF-IDF":
    
    tfidf = TfidfVectorizer(utils.prep_process_tokenize).fit_transform(data["description"])
    query_index = utils.query_idx(data, CO_NAME)
    similar_co_ind, cosine_sims = utils.query_sim(tfidf, query_index, TOP_K)
    tfidf_result = utils.get_result(experiment, data, similar_co_ind, TOP_K)
 
elif experiment == "LDA":
    
    data_lda = utils.build_tokenized_col(data, stopwords)
    training_set = utils.get_training_data(data_lda)
    
    dictionary = corpora.Dictionary(training_set)
    corpus = utils.get_corpus(training_set, dictionary)
    lda = LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary, alpha=0.0001, chunksize=CHUNK, random_state=42)
    
    docmat_pad, docmat = utils.doc_matrix(lda, corpus, NUM_TOPICS)
    query_index = utils.query_idx(data, CO_NAME)
    new_doc_mat = utils.get_dict2bow(data, lda, dictionary, query_index)
    query_answer = utils.matpad([new_doc_mat], NUM_TOPICS)[0]
    
    most_sim_ids, sims = utils.get_top_k_similar_docs(query_answer, docmat_pad, 2000, entropy, k=5)
    lda_result = utils.get_result(experiment, data, most_sim_ids, TOP_K)
    

 
# %%

coherence_model_lda = models.CoherenceModel(model=lda, texts=training_set, dictionary=dictionary, coherence="u_mass")
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)