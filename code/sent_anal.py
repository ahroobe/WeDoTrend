# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 09:55:39 2019

@author: HPE
"""
#############최종 정리####################

import os
import pandas as pd  #data 분석 라이브러리(data frame), incoding 하는 함수,코딩하는 엑셀
import numpy as np   #

project_dir = 'C:/PySrc/project/'      #파일 위치 지정
df = pd.read_csv(project_dir + 'crawling.csv')
print(df.head(5))
    
df_n = df[df["text"].notnull()] #title & text(secret 제외)
print(np.shape(df_n))

# Set Korean Font
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

# Make dataframe 
df_n_1 = df_n.copy()

df_n_1['접수일자'] = [i[0:10] for i in df_n_1['datetime']]
df_n_1['접수일자'] = pd.to_datetime(df_n_1['접수일자'])
df_n_1['요일'] = df_n_1['접수일자'].dt.dayofweek


voc_time = []
for i in df_n_1['datetime']:
    if len(i) == 21:
        i = (i[:14] + '0' + i[14:])
    else:
        pass

    if i[11:13] == '오전':
        voc_time.append(i[14:])
    else:
        voc_time.append(str(int(i[14:16]) + 12) + i[16:])


df_n_1['접수시각'] = voc_time

df_n_1['년도'] = [i[0:4] for i in df_n_1['datetime']]
df_n_1['월'] = [i[5:7] for i in df_n_1['datetime']]
df_n_1['일'] = [i[8:10] for i in df_n_1['datetime']]
df_n_1['년월'] = df_n_1['년도'] + df_n_1['월']

df_n_1['년도월일'] = [i[0:10] for i in df_n_1['datetime']]   

# Reset index and export file
df_n_2 = df_n_1.reset_index(drop=True)        
df_n_2.to_excel(project_dir + 'df_n_2.xlsx')




################################선율이꺼랑 똑같은지 비교

# Load data 
df_n_3 = pd.read_excel(project_dir + 'syc_df.xlsx')
df_n_3 = df_n_3.set_index('Unnamed: 0')


# Parsing dataset
df_parsing = np.array(df_n_3['title'] + ' ' + df_n_3['text'])  



# Parsing 
from konlpy.tag import Kkma
import time

kkma = Kkma()
'''
def flatten_double_list(mainlist):
    return [item for sublist in mainlist for item in sublist]
'''
# Stop words
stop_w = []

xlsx_file = os.path.join(project_dir, 'df_parsing.xlsx')
writer = pd.ExcelWriter(xlsx_file)

report_docs = np.array(df_parsing)
documents = []
process_time = time.time()    #process time 알기 위해서
for report_doc in report_docs:
    if report_doc is np.nan:
        pass
    else:
        words_tags = kkma.pos(report_doc)
        words = [word+'/'+tag for (word, tag) in words_tags if tag in ['NNG','NNP','NNB','VA']]
        # Remove stopwords
        words = [word for word in words if word not in stop_w]
    
    documents.append(words)
process_time = time.time() - process_time        #process time 계산
print ('Process time: %.3f secs' %(process_time))

# Excel
excel_save = pd.DataFrame(documents)
excel_save.to_excel(writer)   
writer.save()

# data set : df_parsing  

################################pol_anal##########################################

#감성사전 입력
import pandas as pd  
pol = pd.read_csv(project_dir + 'polarity.csv')




#긍정일 확률 50% 이상인 감성사전
pos_dict = {}

for i in range(len(pol)):
    pos_dict[pol['ngram'][i]] = pol['POS'][i]
    
pos_dict1 = {key: value for key, value in pos_dict.items() if value >= 0.5}
print(min(pos_dict1.values()))


#부정일 확률 50% 이상인 감성사전
neg_dict={}

for i in range(len(pol)):
    neg_dict[pol['ngram'][i]] = pol['NEG'][i]     

neg_dict1 = {key: value for key, value in neg_dict.items() if value >= 0.5}
print(min(neg_dict1.values()))
   


# 분석 시작
pol_anal = pd.read_excel(project_dir + 'df_parsing.xlsx')



pos_lst={}
for i in range(len(pol_anal)):
    pos_count=0
    for j in range(len(pol_anal.loc[i])-1):
        if pol_anal.loc[i][j] is np.nan:
            break
        elif pol_anal.loc[i][j] in pos_dict1.keys():
            pos_count+=1
    pos_lst.setdefault(i,pos_count)


neg_lst={}
for i in range(len(pol_anal)):
    neg_count=0
    for j in range(len(pol_anal.loc[i])-1):
        if pol_anal.loc[i][j] is np.nan:
            break
        elif pol_anal.loc[i][j] in neg_dict1.keys():
            neg_count+=1
    neg_lst.setdefault(i,neg_count) 




# 감성분석 결과 dataframe 만들기
alls = []

for key in pos_lst.keys():
    new = {"pos":pos_lst[key],
       "neg":neg_lst[key],
       "sent_diff":pos_lst[key]-neg_lst[key],
       "tot":pos_lst[key]+neg_lst[key],
       }
    alls.append(new)
    
sent_df = pd.DataFrame(alls)

def div(a,b):
    if b!=0:
        return a/b
    else:
        return 0
    
    
sent_df["sent"]=sent_df.apply(lambda x: div(x["sent_diff"],x["tot"]),axis=1)
sent_df.head()


def p_n(a):
    if a>0:
        return "긍정"
    elif a<0:
        return "부정"
    else:
        return "중립"
    
    
sent_df["sent_total"]=sent_df.apply(lambda x: p_n(x["sent"]),axis=1)


# 결과 파일에 시간 데이터 삽입

sent_df['year']=df_n_3['년도']
sent_df['month']=df_n_3['월']
sent_df['day']=df_n_3['일']
sent_df['ym']=df_n_3['년월']
sent_df['ymd']=df_n_3['년도월일']
sent_df['dw']=df_n_3['요일']
sent_df['time']=df_n_3['접수시각']


# Excel
xlsx_file_sent = os.path.join(project_dir, 'sent_final.xlsx')
writer_sent = pd.ExcelWriter(xlsx_file_sent)
excel_save_sent = pd.DataFrame(sent_df)
excel_save_sent.to_excel(writer_sent)   
writer_sent.save()