# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# Importing Gensim
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt

#import documents
import os

# Set Korean Font
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

project_dir = 'C:/Users/HPE/Downloads/VOC/'
data_path_1 = os.path.join(project_dir, 'n_total.xlsx')
data_paths = [data_path_1]

#Load data
for i in range(len(data_paths)):
    
    data_path = data_paths[i]
    
    data_path = data_paths[0]
    data = pd.read_excel(data_path, encoding='cp949')
    data = data.drop(['Unnamed: 0'], axis = 1)
    
    tmp = data.values.tolist()
    
    documents = []
    for i in range(len(tmp)):
        documents.append([x for x in tmp[i] if str(x) != 'nan'])
    
    # Creating the term dictionary of our courpus, where every unizue term is assigned an index
    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    print(len(dictionary))
    
    # Converting list of documnets(corpus) into Document Term Matrix using dictionary prepared above
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in documents]
    Lda = gensim.models.ldamodel.LdaModel
    
    # Check the coherence value to find the best K
    def compute_coherence_values(dictionary, corpus, texts, start, limit, step):
        """
        Compute c_v coherence for various number of topics
         
        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics
         
        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """    
        coherence_values = []
        model_list = []
        
        for num_topics in range(start, limit, step):
            model = Lda(doc_term_matrix, num_topics = num_topics, id2word = dictionary, passes=50)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
            
        return model_list, coherence_values
       
    model_list, coherence_values = compute_coherence_values(dictionary, corpus, documents, 2, 10, 1)
    
    x = range(2, 10, 1)
    for m, cv in zip(x, coherence_values):
        print("Num Topics = ", m, "has Coherence Value of ", round(cv, 4))
    
    
    best_K = 0
    for m in x:
        if coherence_values[m] - coherence_values[m-1] > 0:
            best_K += (m + 2)
            break
        else:
            pass
    
    print(best_K)
         
            
