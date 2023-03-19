import argparse

from gensim.models import LdaModel
from gensim import models, corpora
from gensim.test.utils import datapath
import nltk


from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import entropy

import pandas as pd
import utils


def main(args):
    
    # nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('english')


    data = pd.read_csv("data\\co_data.csv")

    # arguments
    EXP = args.experiment
    CO_NAME = args.co_name
    TOP_K = args.top_k
    NUM_TOPICS = args.num_topics
    CHUNK = args.chunks
    ALPHA = args.alpha
    PASSES = args.passes
    ITERATIONS = args.iterations
    

    if EXP == "TF-IDF":
        
        vectorizer = TfidfVectorizer(preprocessor=utils.pre_process, ngram_range=(1,2))
        tfidf = vectorizer.fit_transform(data["description"])
        query_index = utils.query_idx(data, CO_NAME)
        similar_co_ind, cosine_sims = utils.query_sim(tfidf, query_index, TOP_K)
        tfidf_result = utils.get_result_tfidf(data, similar_co_ind, cosine_sims, TOP_K)
        utils.save_tf_idf_scores(tfidf, vectorizer, query_index)
        
    elif EXP == "LDA":
        
        # ext_stopwords = utils.extend_stopwords(stopwords)
        data_lda = utils.build_tokenized_col(data)
        training_set = utils.get_training_data(data_lda)
        
        dictionary = corpora.Dictionary(training_set)
        corpus = utils.get_corpus(training_set, dictionary)
        lda = LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary, chunksize=CHUNK, iterations=ITERATIONS, passes= PASSES, random_state=42)
        
        docmat_pad, docmat = utils.doc_matrix(lda, corpus, NUM_TOPICS)
        query_index = utils.query_idx(data, CO_NAME)
        new_doc_mat = utils.get_dict2bow(data, lda, dictionary, query_index)
        query_answer = utils.matpad([new_doc_mat], NUM_TOPICS)[0]
        
        most_sim_ids, sims = utils.get_top_k_similar_docs(query_answer, docmat_pad, 2000, entropy, k=5)
        lda_result = utils.get_result_lda(data, most_sim_ids)
        
        coherence_model_lda = models.CoherenceModel(model=lda, texts=training_set, dictionary=dictionary, coherence="u_mass")
        coherence_lda = coherence_model_lda.get_coherence()
        print("\nCoherence Score: ", coherence_lda)
        
        lda.save("model\\lda.model")
        
        
        

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--experiment", default="LDA", type=str)
    parser.add_argument("--co_name", default="Vahanalytics", type=str)
    parser.add_argument("--top_k", default=5, type=int)
    parser.add_argument("--num_topics", default=4, type=int)
    parser.add_argument("--chunk", default=2000, type=int)
    parser.add_argument("--alpha", default="auto", type=str)
    parser.add_argument("--iterations", default=1000, type=int)
    parser.add_argument("--passes", default=10, type=int)
    
    args = parser.parse_args()
    
    main(args)

    
