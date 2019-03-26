# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas as pd
import numpy as np

project_dir = 'C:/Users/HPE/Downloads/VOC/'
df = pd.read_csv(project_dir + 'crawling.csv')
print(df.head(5))

df_s = df[df["text"].isnull()] # title only
print(np.shape(df_s))

df_n = df[df["text"].notnull()]
print(np.shape(df_n))

'''
# 비밀글일 때
tmp = np.array(df_s)
s_title = tmp[:,1]
print(s_title[0])
'''

# 비밀글 아닐 때
n_total = np.array(df_n['title'] + ' ' + df_n['text'])
print(n_total[0])


# Set Korean Font
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)


from konlpy.tag import Kkma
import time

kkma = Kkma()


# Stop words
stop_w = ['것', '수', '같', '때', '거']

xlsx_file = os.path.join(project_dir, 'n_total.xlsx')
writer = pd.ExcelWriter(xlsx_file)

report_docs = np.array(n_total)
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



'''
# Naive bayes classifier + smoothing
def naive_bayes_classifier(test, train, all_count):
    counter = 0
    list_count = []
    for i in test:
        for j in range(len(train)):
            if i == train[j]:
                counter = counter + 1
        list_count.append(counter)
        counter = 0
    list_naive = []
    for i in range(len(list_count)):
        list_naive.append((list_count[i]+1)/float(len(train)+all_count))
    result = 1
    for i in range(len(list_naive)):
        result *= float(round(list_naive[i], 6))
    return float(result)*float(1.0/3.0)
'''