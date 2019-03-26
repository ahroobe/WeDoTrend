# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 20:02:54 2018

@author: Gonie
"""

# Import Modules
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
from konlpy.tag import Kkma
import nltk
from wordcloud import WordCloud

import time
 
# Set project and data directory ----------------------------------------------
project_dir = 'C:/Users/HPE/Downloads/VOC/'
df = pd.read_csv(project_dir + 'crawling.csv')
print(df.head(5))

df_s = df[df["text"].isnull()] # title only
print(np.shape(df_s))

df_n = df[df["text"].notnull()]
print(np.shape(df_n))

# Set Korean Font
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)


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
df['년도월일'] = [i[0:10] for i in df['datetime']]

df['년월'] = df['년도'] + df['월']

# New column
# 접수일자, 요일, 접수시각, 년도, 월

    
# EDA Functions
def pivot(data, values, index, columns, aggfunc):
    total_columns = []
    if values is not None:
        total_columns = total_columns + values
    if index is not None:
        total_columns = total_columns + index
    if columns is not None:
        total_columns = total_columns + columns
    total_columns = list(set(total_columns))
    data = data[total_columns]
    result = pd.pivot_table(data, values=values, index=index, columns=columns, aggfunc=aggfunc)
    return result
     
def freq_to_ratio(data, axis):
    summation = data.sum(axis)
    result = data.div(summation, axis=1-axis)
    return result
 
def plot_counts(data, column, sort=False):
    if sort:
        data_counts = data[column].value_counts().sort_index()
    else:
        data_counts = data[column].value_counts()
    plt.figure(figsize=(10, 8))
    plt.bar(range(len(data_counts)), data_counts)
    plt.xticks(range(len(data_counts)), data_counts.index, rotation=90)
    plt.show()
 


# 1: 시계열 ===================================================================

DF = df.copy()

# Load index
with open(project_dir + 'TOPIC/CONTENT/idx_list.txt', "r") as file:
    idx_list = eval(file.readline())
 
process_time = time.time()
 
for k in range(len(idx_list)):
    DF_sub = DF.iloc[idx_list[k]]
     
    xlsx_file = os.path.join(project_dir, 'TOPIC/CONTENT/TOPIC_XLSX/시계열_{}.xlsx'.format(k))
    writer = pd.ExcelWriter(xlsx_file)
    
    # Save Table to Excel            
    pivot_table = DF_sub.groupby('년월').size()
    pivot_table = pivot_table.sort_values(ascending=False)
    pivot_table.to_excel(writer)
    
    data_counts = DF_sub['년월'].value_counts().sort_index()
    fig = plt.figure(figsize=(20, 8))
    plt.bar(range(len(data_counts)), data_counts)    
    for j in range(len(data_counts)):
        plt.text(range(len(data_counts))[j]-0.1, data_counts[j]+0.5, data_counts[j])
    plt.xticks(range(len(data_counts)), data_counts.index, rotation=90)
    #save the figure
    fig_name = os.path.join(project_dir, 'TOPIC/CONTENT/TOPIC_PNG/시계열_{}.png'.format(k))
    fig.savefig(fig_name)
    plt.show()            
     
    writer.save()
     
process_time = time.time() - process_time
print ('Process time: %.3f secs' %(process_time))

# 1-1: 일별 ===================================================================

DF = df.copy()

# Load index
with open(project_dir + 'TOPIC/CONTENT/idx_list.txt', "r") as file:
    idx_list = eval(file.readline())
 
process_time = time.time()
 
for k in range(len(idx_list)):
    DF_sub = DF.iloc[idx_list[k]]
     
    xlsx_file = os.path.join(project_dir, 'TOPIC/CONTENT/TOPIC_XLSX/년도월일별_{}.xlsx'.format(k))
    writer = pd.ExcelWriter(xlsx_file)
    
    # Save Table to Excel
    data_counts = DF_sub['년도월일'].value_counts().sort_index()
    data_counts.to_excel(writer)

    writer.save()
     
process_time = time.time() - process_time
print ('Process time: %.3f secs' %(process_time))


# 2: 요일 ===================================================================== 
# Load index
with open(project_dir + 'TOPIC/CONTENT/idx_list.txt', "r") as file:
    idx_list = eval(file.readline())
 
process_time = time.time()
 
for k in range(len(idx_list)):
    DF_sub = DF.iloc[idx_list[k]]
     
    xlsx_file = os.path.join(project_dir, 'TOPIC/CONTENT/TOPIC_XLSX/요일_{}.xlsx'.format(k))
    writer = pd.ExcelWriter(xlsx_file)
    
    # Save Table to Excel            
    pivot_table = DF_sub.groupby('요일').size()
    pivot_table = pivot_table.sort_values(ascending=False)
    pivot_table.to_excel(writer)
    
    data_counts = DF_sub['요일'].value_counts().sort_index()
    fig = plt.figure(figsize=(10, 8))
    plt.bar(range(len(data_counts)), data_counts)    
    for j in range(len(data_counts)):
        plt.text(range(len(data_counts))[j]-0.1, data_counts[j]+0.5, data_counts[j])
    plt.xticks(range(len(data_counts)), data_counts.index, rotation=90)
    #save the figure
    fig_name = os.path.join(project_dir, 'TOPIC/CONTENT/TOPIC_PNG/요일_{}.png'.format(k))
    fig.savefig(fig_name)
    plt.show()            
     
    writer.save()
     
process_time = time.time() - process_time
print ('Process time: %.3f secs' %(process_time))


# 3: 시간대 ===================================================================
# Load index
with open(project_dir + 'TOPIC/CONTENT/idx_list.txt', "r") as file:
    idx_list = eval(file.readline())
 
process_time = time.time()
 
for k in range(len(idx_list)):
    
    DF_sub = DF.iloc[idx_list[k]]
     
    xlsx_file = os.path.join(project_dir, 'TOPIC/CONTENT/TOPIC_XLSX/시간대_{}.xlsx'.format(k))
    writer = pd.ExcelWriter(xlsx_file)
    
    # Save Table to Excel 
    DF_sub['접수시간대'] = [t[0:2] for t in DF_sub['접수시각']]
    pivot_table = DF_sub.groupby('접수시간대').size()
    pivot_table = pivot_table.sort_values(ascending=False)
    pivot_table.to_excel(writer)
    
    data_counts = DF_sub['접수시간대'].value_counts().sort_index()
    fig = plt.figure(figsize=(20, 8))
    plt.bar(range(len(data_counts)), data_counts)    
    for j in range(len(data_counts)):
        plt.text(range(len(data_counts))[j]-0.1, data_counts[j]+0.5, data_counts[j])
    plt.xticks(range(len(data_counts)), data_counts.index, rotation=90)
    #save the figure
    fig_name = os.path.join(project_dir, 'TOPIC/CONTENT/TOPIC_PNG/시간대_{}.png'.format(k))
    fig.savefig(fig_name)
    plt.show()            
     
    writer.save()
     
process_time = time.time() - process_time
print ('Process time: %.3f secs' %(process_time))
    

# 4: Word Cloud ===============================================================

topk = 100
kkma = Kkma()

def flatten_double_list(mainlist):
    return [item for sublist in mainlist for item in sublist]


n_total = df_n['title'] + ' ' + df_n['text']

# Load index
with open(project_dir + 'TOPIC/CONTENT/idx_list.txt', "r") as file:
    idx_list = eval(file.readline())
 

stop_w = ['것', '수', '같', '때', '거']

for k in range(len(idx_list)):
    process_time = time.time()
    DF_sub = n_total.iloc[idx_list[k]]
  
    xlsx_file = os.path.join(project_dir, 'TOPIC/CONTENT/TOPIC_XLSX/단어빈도수_{}.xlsx'.format(k))
    writer = pd.ExcelWriter(xlsx_file)
    
    # Get Noun V2
    report_docs = np.array(DF_sub)
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
        
    documents = flatten_double_list(documents)
    
    process_time = time.time() - process_time
    print ('Process time: %.3f secs' %(process_time))
    
    # Word frequency
    word_freq_topk = nltk.FreqDist(documents).most_common(topk)
    names = [item[0] for item in word_freq_topk]
    counts = [item[1] for item in word_freq_topk]
    # WordCloud
    wordcloud = WordCloud(font_path="C:/Windows/Fonts/malgun.ttf",
                          relative_scaling=0.2,
                          background_color='white').generate_from_frequencies(dict(word_freq_topk))
    plt.figure(figsize=(30, 30))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    fig_name = os.path.join(project_dir, 'TOPIC/CONTENT/TOPIC_PNG/워드클라우드_{}.png'.format(k))
    plt.savefig(fig_name)
    plt.close()
     
    # Excel
    excel_save = pd.DataFrame(word_freq_topk, columns=['단어', '빈도수'])
    excel_save.to_excel(writer)        
    writer.save()


# 5: Assigned docs ============================================================

# Load index
with open(project_dir + 'TOPIC/CONTENT/idx_list.txt', "r") as file:
    idx_list = eval(file.readline())
 

for k in range(len(idx_list)):
    process_time = time.time()
    DF_sub = n_total.iloc[idx_list[k]]
  
    xlsx_file = os.path.join(project_dir, 'TOPIC/CONTENT/TOPIC_XLSX/할당문서_{}.xlsx'.format(k))
    writer = pd.ExcelWriter(xlsx_file)
    
    DF_sub.to_excel(writer)
    writer.save()   