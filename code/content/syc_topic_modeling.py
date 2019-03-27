# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 09:38:01 2019

@author: HPE
"""

import os
import pandas as pd
import numpy as np

from konlpy.tag import Kkma
import time

import gensim
from gensim import corpora

import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt

import nltk
from wordcloud import WordCloud

# 1. Preprocessing ============================================================
# Load data
project_dir = 'C:/Users/HPE/Downloads/VOC/'
df = pd.read_csv(project_dir + 'crawling.csv')
print(df.head(5))

df_n = df[df["text"].notnull()]
print(np.shape(df_n))

# Set Korean Font
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

# Make dataframe
df = df_n.copy()

df['접수일자'] = [i[0:10] for i in df['datetime']]
df['접수일자'] = pd.to_datetime(df['접수일자'])
df['요일'] = df['접수일자'].dt.dayofweek


df['접수시각'] = df['datetime'].copy()

voc_time = []
for i in df['datetime']:
    if len(i) == 21:
        i = (i[:14] + '0' + i[14:])
    else:
        pass

    if i[11:13] == '오전':
        voc_time.append(i[14:])
    else:
        voc_time.append(str(int(i[14:16]) + 12) + i[16:])


df['접수시각'] = voc_time

df['년도'] = [i[0:4] for i in df['datetime']]
df['월'] = [i[5:7] for i in df['datetime']]
df['일'] = [i[8:10] for i in df['datetime']]
df['년월'] = df['년도'] + df['월']

df['년도월일'] = [i[0:10] for i in df['datetime']]   

# Reset index and export file
df = df.reset_index(drop=True)
df.to_excel(project_dir + 'syc_df.xlsx')




# 2. Define functions =========================================================
# Load data
df = pd.read_excel(project_dir + 'syc_df.xlsx')
df = df.set_index('Unnamed: 0')


def syc_tm(df, start, end, n_topic):

    df = df[(df['년도월일'] >= start) & (df['년도월일'] < end)]
    
    # Topic Modeling ==========================================================
    def pre_LDA(dat):   
        tmp = dat.values.tolist()
         
        docs = []
        for i in range(len(tmp)):
            docs.append([x for x in tmp[i] if str(x) != 'nan'])
         
        # Creating the term dictionary of our courpus, where every unique term is assigned an index
        dictionary = corpora.Dictionary(docs)
        corpus = [dictionary.doc2bow(doc) for doc in docs]   
        # Converting list of documnets(corpus) into Document Term Matrix using dictionary prepared above
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs]
         
        return docs, dictionary, corpus, doc_term_matrix
    
    data = (df['title'] + ' ' + df['text'])
    
    # POS tagging
    kkma = Kkma()
    # Stop words list
    stop_w = ['것', '수', '같', '때', '거']
    
    xlsx_file = os.path.join(project_dir, 'syc_documents.xlsx')
    writer = pd.ExcelWriter(xlsx_file)
    
    report_docs = np.array(data)
    documents = []
    process_time = time.time()
    for report_doc in report_docs:
        if report_doc is np.nan:
            pass
        else:
            words_tags = kkma.pos(report_doc)
            words = [word for (word, tag) in words_tags if tag in ['NNG','NNP','NNB', 'VA']]
            # Remove stopwords
            words = [word for word in words if word not in stop_w]
        
        documents.append(words)
    process_time = time.time() - process_time
    print ('Process time: %.3f secs' %(process_time))
    
    # Excel
    excel_save = pd.DataFrame(documents)
    excel_save.to_excel(writer)   
    writer.save()
    
    
    # Load model
    Lda = gensim.models.ldamodel.LdaModel
    
    documents1 = pd.DataFrame(documents)
    documents1.fillna(value=pd.np.nan, inplace=True)
    
    docs, dictionary, corpus, doc_term_matrix = pre_LDA(documents1)
    ldamodel = Lda(corpus = doc_term_matrix, id2word = dictionary, num_topics = n_topic, passes=50, minimum_probability=0)
    
    # Visualize the topics
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
    pyLDAvis.save_html(vis, project_dir + 'TOPIC/CONTENT/LDA/LDAvis.html')
     
    # Assign docs to topic
    lda_corpus = [max(prob,key = lambda y:y[1]) for prob in ldamodel[corpus]]
    #lda_corpus[0]
     
    case_list = [[] for i in range(n_topic)]
    idx_list = [[] for i in range(n_topic)]
     
    for i, x in enumerate(lda_corpus):
        case_list[x[0]].append(docs[i])
        idx_list[x[0]].append(i)
    
    # Save case list and index list
    with open(project_dir + 'TOPIC/CONTENT/case_list_{}_{}.txt'.format(start, end), "w") as file:
        file.write(str(case_list))
     
    with open(project_dir + 'TOPIC/CONTENT/idx_list_{}_{}.txt'.format(start, end), "w") as file:
        file.write(str(idx_list))
    
    # Word Cloud ==============================================================
    topk = 100
    
    def flatten_double_list(mainlist):
        return [item for sublist in mainlist for item in sublist]
    
    
    for k in range(len(idx_list)):
        process_time = time.time()
        
        # Assign docs
        xlsx_file_doc = os.path.join(project_dir, 'TOPIC/CONTENT/TOPIC_XLSX/할당문서_{}_{}_{}.xlsx'.format(start, end, k))
        writer_doc = pd.ExcelWriter(xlsx_file_doc)
        DF_sub = documents1.iloc[idx_list[k]]
    
        DF_sub.to_excel(writer_doc)
        writer_doc.save()
    
    
        # Word frequency  
        xlsx_file = os.path.join(project_dir, 'TOPIC/CONTENT/TOPIC_XLSX/단어빈도수_{}_{}_{}.xlsx'.format(start, end, k))
        writer = pd.ExcelWriter(xlsx_file)
    
        words_pot = list(np.asarray(DF_sub))
        words_pot = [word for word in flatten_double_list(words_pot) if str(word) != 'nan']
        word_freq_topk = nltk.FreqDist(words_pot).most_common(topk)

        # WordCloud
        wordcloud = WordCloud(font_path="C:/Windows/Fonts/malgun.ttf",
                              relative_scaling=0.2,
                              background_color='white').generate_from_frequencies(dict(word_freq_topk))
        plt.figure(figsize=(30, 30))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        fig_name = os.path.join(project_dir, 'TOPIC/CONTENT/TOPIC_PNG/워드클라우드_{}_{}_{}.png'.format(start, end, k))
        plt.savefig(fig_name)
        plt.close()
         
        excel_save = pd.DataFrame(word_freq_topk, columns=['단어', '빈도수'])
        excel_save.to_excel(writer)        
        writer.save()
    
        process_time = time.time() - process_time
        print ('Process time: %.3f secs' %(process_time))
    
