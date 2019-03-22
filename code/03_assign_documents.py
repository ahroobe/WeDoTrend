# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 20:01:24 2018

@author: Gonie
"""

 
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
project_dir = 'C:/Users/HPE/Downloads/VOC/'
data_path_1 = os.path.join(project_dir, 's_title.xlsx')
data_paths = [data_path_1]

print(data_paths)
 
# Set Korean Font
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
 
 
def pre_LDA(data):   
    tmp = data.values.tolist()
     
    documents = []
    for i in range(len(tmp)):
        documents.append([x for x in tmp[i] if str(x) != 'nan'])
     
    # Creating the term dictionary of our courpus, where every unizue term is assigned an index
    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]   
    # Converting list of documnets(corpus) into Document Term Matrix using dictionary prepared above
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in documents]
     
    return documents, dictionary, corpus, doc_term_matrix
 
'''     
def load_data_LDA(data_paths, contents, data_types):
    for data_path in data_paths:
  
        data_name = os.path.split(data_path)[1]
        _, content, data_type = data_name.replace('.xlsx', '').split('_')
        if content == contents and data_type == data_types:
            data = pd.read_excel(data_path, encodeing='cp949')
 
            return data
'''
 
def plot_counts_per_topic(idx_lists):
    xs = np.arange(len(idx_lists))
    ys = [len(idx_lists[i]) for i in range(len(idx_lists))]
       
    width = 0.5
     
    fig = plt.figure(figsize=(10, 8))
    plt.bar(xs, ys, width)
    for j in range(len(xs)):
        plt.text(range(len(xs))[j]-0.05, ys[j]+1, ys[j], fontsize=12)
    plt.xticks(range(len(xs)))
     
    # Save the figure
    fig_name = os.path.join(project_dir, 'TOPIC/PLOT/plot.png')
    fig.savefig(fig_name)
    plt.show()
 
# Load model
Lda = gensim.models.ldamodel.LdaModel
 
data_contents = ['신고내용', '종결내용']
data_types = ['가정폭력', '데이트폭력', '몰카', '성폭력']
 
best_Ks_1 = [8, 6, 3, 5]
best_Ks_2 = [6, 4, 3, 4]
 
dic_Ks_1 = {}
for i in range(len(data_types)):
    dic_Ks_1[data_types[i]] = best_Ks_1[i]
 
dic_Ks_2 = {}
for i in range(len(data_types)):
    dic_Ks_2[data_types[i]] = best_Ks_2[i]
 

# Load data
data = pd.read_excel(data_paths[0], encodeing='cp949')
data = data.drop(['Unnamed: 0'], axis = 1)

best_K = 6

# Modeling and get the result
documents, dictionary, corpus, doc_term_matrix = pre_LDA(data)
ldamodel = Lda(corpus = doc_term_matrix, id2word = dictionary, num_topics = best_K, passes=50, minimum_probability=0)

# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
pyLDAvis.save_html(vis, project_dir + 'TOPIC/LDA/LDAvis.html')
 
# Assign documents to topic
lda_corpus = [max(prob,key = lambda y:y[1]) for prob in ldamodel[corpus]]
lda_corpus[0]
 
case_lists = [[] for i in range(best_K)]
idx_lists = [[] for i in range(best_K)]
 
for i, x in enumerate(lda_corpus):
    case_lists[x[0]].append(documents[i])
    idx_lists[x[0]].append(i)

# Get the histogram
plot_counts_per_topic(idx_lists, data_content, data_type)
 
with open(project_dir + '0905/TOPIC/TOPIC_case/case_list_{}_{}.txt'.format(data_content, data_type), "w") as file:
    file.write(str(case_lists))
 
with open(project_dir + '0905/TOPIC/TOPIC_idx/idx_list_{}_{}.txt'.format(data_content, data_type), "w") as file:
    file.write(str(idx_lists))



'''
for data_content in data_contents:
    for data_type in data_types:               
        
        # Load data
        data = load_data_LDA(data_paths, data_content, data_type)       
        if data_content == '신고내용':
            best_K = dic_Ks_1[data_type]
        else:
            best_K = dic_Ks_2[data_type]
    
        # Modeling and get the result
        documents, dictionary, corpus, doc_term_matrix = pre_LDA(data)
        ldamodel = Lda(corpus = doc_term_matrix, id2word = dictionary, num_topics = best_K, passes=50, minimum_probability=0)
        
        # Visualize the topics
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
        pyLDAvis.save_html(vis, project_dir + '0905/TOPIC/TOPIC_PNG/LDAvis_{}_{}.html'.format(data_content, data_type))
 
        # Assign documents to topic
        lda_corpus = [max(prob,key = lambda y:y[1]) for prob in ldamodel[corpus]]
        lda_corpus[0]
         
        case_lists = [[] for i in range(best_K)]
        idx_lists = [[] for i in range(best_K)]
         
        for i, x in enumerate(lda_corpus):
            case_lists[x[0]].append(documents[i])
            idx_lists[x[0]].append(i)
        
        # Get the histogram
        plot_counts_per_topic(idx_lists, data_content, data_type)
         
        with open(project_dir + '0905/TOPIC/TOPIC_case/case_list_{}_{}.txt'.format(data_content, data_type), "w") as file:
            file.write(str(case_lists))
         
        with open(project_dir + '0905/TOPIC/TOPIC_idx/idx_list_{}_{}.txt'.format(data_content, data_type), "w") as file:
            file.write(str(idx_lists))
         
         
        print('{}_{}_done\n'.format(data_content, data_type))
 