#!/usr/bin/env python
# coding: utf-8

# In[1]:


import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import datetime
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import json
init_notebook_mode(connected=True)
import pandas as pd
from flask import Flask, render_template, request

from collections import defaultdict
from konlpy.tag import Kkma
import time
import pickle
import gensim
from gensim import corpora

import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt

import nltk
from wordcloud import WordCloud
import numpy as np
import os
# Set Korean Font
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)


# # syc_df.xlsx
# * input: crawling.csv (in https://github.com/ahroobe/WeDoTrend/blob/master/code/crawling.csv)
# * crawling code: https://github.com/ahroobe/WeDoTrend/blob/master/code/Crawler.ipynb
# 
# ```
# 
# # 1. Preprocessing ============================================================
# # Load data
# project_dir = 'C:/Users/HPE/Downloads/VOC/'
# df = pd.read_csv(project_dir + 'crawling.csv')
# print(df.head(5))
# 
# df_n = df[df["text"].notnull()]
# print(np.shape(df_n))
# 
# # Set Korean Font
# from matplotlib import font_manager, rc
# font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf").get_name()
# rc('font', family=font_name)
# 
# # Make dataframe
# df = df_n.copy()
# 
# df['접수일자'] = [i[0:10] for i in df['datetime']]
# df['접수일자'] = pd.to_datetime(df['접수일자'])
# df['요일'] = df['접수일자'].dt.dayofweek
# 
# 
# df['접수시각'] = df['datetime'].copy()
# 
# voc_time = []
# for i in df['datetime']:
#     if len(i) == 21:
#         i = (i[:14] + '0' + i[14:])
#     else:
#         pass
# 
#     if i[11:13] == '오전':
#         voc_time.append(i[14:])
#     else:
#         voc_time.append(str(int(i[14:16]) + 12) + i[16:])
# 
# 
# df['접수시각'] = voc_time
# 
# df['년도'] = [i[0:4] for i in df['datetime']]
# df['월'] = [i[5:7] for i in df['datetime']]
# df['일'] = [i[8:10] for i in df['datetime']]
# df['년월'] = df['년도'] + df['월']
# 
# df['년도월일'] = [i[0:10] for i in df['datetime']]   
# 
# # Reset index and export file
# df = df.reset_index(drop=True)
# df.to_excel(project_dir + 'syc_df.xlsx')
#     
# ```
# # syc_document.xlsx
# * input (df): sys_df.xlsx
# 
# ```
# df = df[(df['년도월일'] >= start) & (df['년도월일'] < end)]
# st_idx = df.index[0]
# en_idx = df.index[-1]
# # Topic Modeling ==========================================================
# def pre_LDA(dat):   
#     tmp = dat.values.tolist()
# 
#     docs = []
#     for i in range(len(tmp)):
#         docs.append([x for x in tmp[i] if str(x) != 'nan'])
# 
#     # Creating the term dictionary of our courpus, where every unique term is assigned an index
#     dictionary = corpora.Dictionary(docs)
#     corpus = [dictionary.doc2bow(doc) for doc in docs]   
#     # Converting list of documnets(corpus) into Document Term Matrix using dictionary prepared above
#     doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs]
# 
#     return docs, dictionary, corpus, doc_term_matrix
# 
# data = (df['title'] + ' ' + df['text'])
# 
# # POS tagging
# kkma = Kkma()
# # Stop words list
# stop_w = ['것', '수', '같', '때', '거']
# 
# xlsx_file = os.path.join(project_dir, 'syc_documents.xlsx')
# writer = pd.ExcelWriter(xlsx_file)
# 
# report_docs = np.array(data)
# documents = []
# process_time = time.time()
# for report_doc in report_docs:
#     if report_doc is np.nan:
#         pass
#     else:
#         words_tags = kkma.pos(report_doc)
#         words = [word for (word, tag) in words_tags if tag in ['NNG','NNP','NNB', 'VA']]
#         # Remove stopwords
#         words = [word for word in words if word not in stop_w]
# 
#     documents.append(words)
# process_time = time.time() - process_time
# print ('Process time: %.3f secs' %(process_time))
# 
# # Excel
# excel_save = pd.DataFrame(documents)
# excel_save.to_excel(writer)   
# writer.save()
# ```
# 

# # 데이터 전처리

# In[10]:


df = pd.read_excel('./dataset/syc_df.xlsx')
assign_list = []
for i in range(0,5):
    a = pd.read_excel('./content/TOPIC_XLSX/할당문서_%s.xlsx'%i)
    assign_list.append(list(a.index))
    
key = dict()
for i in range(0,5):
    key[i] = pd.read_excel('./content/TOPIC_XLSX/단어빈도수_%s.xlsx'%i)[:5]

hj_graph_list=[]
for i in range(0,5):
    df1 = df[df.index.isin(assign_list[i])]
    count = df1.groupby(['접수일자']).size().reset_index(name="count")
    hj_graph_list.append(count)
    
df['시간'] =df['접수시각'].apply(lambda x: x[:2]) 
df['월'] = df['datetime'].apply(lambda x:x[:7])
hj_graph = df.groupby(['월']).size().reset_index(name="count")

document = pd.read_excel('./dataset/syc_documents.xlsx', encoding='cp949')
document.fillna(value=pd.np.nan, inplace=True)

donut_list = []

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


pickle_list = []
k = 0
for i in range(0,5):
    name = './content/TOPIC_XLSX/날짜별키워드_{}.pkl'.format(i)
    dat = load_obj(name)
    pickle_list.append(dat)

sent = pd.read_excel('./dataset/sent_final.xlsx')


# # modules

# In[3]:


def make_keyword(datetime):
    m_key = []
    datetime = datetime[:8]
    for i in range(0,5):
        dictionary = defaultdict(lambda:0)
        for date in pickle_list[i].keys():
            if datetime in date:
                for k in pickle_list[i][date]:
                    dictionary[k[0]] += k[1]
        a = sorted(dictionary.items(), key=lambda k_v: k_v[1], reverse=True)
        m_key.append(a[:5])
    return m_key


# In[4]:


def syc_tm(df, start, end, n_topic):
    dirs = "%s_%s_%s"%(start,end,n_topic)
    
    try:
        os.stat('./content/%s'%dirs)
        return -1
    except:
        os.mkdir('./content/%s'%dirs)
        os.mkdir('./content/%s/TOPIC_PNG/'%dirs)
        os.mkdir('./content/%s/TOPIC_XLSX/'%dirs)
        os.mkdir('./static/img/%s/'%dirs)
   
    df = df[(df['년도월일'] >= start) & (df['년도월일'] < end)]
    st_idx = df.index[0]
    en_idx = df.index[-1]
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
    
    
    # Load model
    Lda = gensim.models.ldamodel.LdaModel
    

    documents1 = document[document.index.isin(range(st_idx,en_idx+1))]
    docs, dictionary, corpus, doc_term_matrix = pre_LDA(documents1)
    ldamodel = Lda(corpus = doc_term_matrix, id2word = dictionary, num_topics = n_topic, passes=50, minimum_probability=0)
    
    # Visualize the topics
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
    pyLDAvis.save_html(vis, './content/%s/LDAvis.html'%dirs)
     
    # Assign docs to topic
    lda_corpus = [max(prob,key = lambda y:y[1]) for prob in ldamodel[corpus]]
    #lda_corpus[0]
     
    case_list = [[] for i in range(n_topic)]
    idx_list = [[] for i in range(n_topic)]
     
    for i, x in enumerate(lda_corpus):
        case_list[x[0]].append(docs[i])
        idx_list[x[0]].append(i)
    
    # Save case list and index list
    with open( './content/{}/case_list.txt'.format(dirs), "w") as file:
        file.write(str(case_list))
     
    with open( './content/{}/idx_list.txt'.format(dirs), "w") as file:
        file.write(str(idx_list))
    
    # Word Cloud ==============================================================
    topk = 100
    
    def flatten_double_list(mainlist):
        return [item for sublist in mainlist for item in sublist]
    
    
    for k in range(len(idx_list)):
        process_time = time.time()
        
        # Assign docs
        xlsx_file_doc = os.path.join( './content/{}/TOPIC_XLSX/할당문서_{}.xlsx'.format(dirs, k))
        writer_doc = pd.ExcelWriter(xlsx_file_doc)
        DF_sub = documents1.iloc[idx_list[k]]
    
        DF_sub.to_excel(writer_doc)
        writer_doc.save()
    
    
        # Word frequency  
        xlsx_file = os.path.join('./content/{}/TOPIC_XLSX/단어빈도수_{}.xlsx'.format(dirs, k))
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
        fig_name = os.path.join('./static/img/{}/워드클라우드_{}.png'.format(dirs,k))
        plt.savefig(fig_name)
        plt.close()
         
        excel_save = pd.DataFrame(word_freq_topk, columns=['단어', '빈도수'])
        excel_save.to_excel(writer)        
        writer.save()
    
        process_time = time.time() - process_time
        print ('Process time: %.3f secs' %(process_time))
    return 1
    


# In[5]:



def make_month_num(df):
    ### return data which has month_number
    date_data = pd.to_datetime(df['접수일자'])
    month_num = date_data.dt.strftime('%Y-%m')
    df['month_num'] = month_num
    df_sum_month = df.groupby(['month_num']).sum()
    df_sum_month['month_num']= df_sum_month.index
    df_sum_month.index = list(range(len(df_sum_month)))
    return df_sum_month


def make_week_num(df):
    ### return data which has week_number
    date_data = pd.to_datetime(df['접수일자'])
    week_num = date_data.dt.strftime('%Y-W%U')
    df['week_num'] = week_num
    df_sum_week = df.groupby(['week_num']).sum()
    df_sum_week['week_num']= df_sum_week.index
    df_sum_week.index = list(range(len(df_sum_week)))
    week_num_new = []
    for i in df_sum_week['week_num']:
        week_num_new.append(datetime.datetime.strptime(i + '-1', "%Y-W%W-%w"))
    df_sum_week['week_num_new'] = week_num_new
    return df_sum_week

    week_scatter_plot  = go.Scatter(x=week_df['week_num_new'], y=week_df['count'])
    layout = go.Layout(
        font={
            "family": "뫼비우스 Regular",
            "size": 15
        },

        title={
            "text": '시간 흐름에 따른 관심도 변화, 주별',
            "font": {
                "size": 24,

            }
        },
        xaxis=dict(
            title='Date',
            titlefont=dict(
                family='뫼비우스 Regular',
                size=18,
                color='#7f7f7f')
        ),
        yaxis=dict(
            title='Count',
            titlefont=dict(
                family='뫼비우스 Regular',
                size=18,
                color='#7f7f7f')
        ),
        hoverlabel={
            "font": {
                "family": "뫼비우스 Regular"}
        }
    )

    fig2 = go.Figure(data=week_scatter_plot, layout=layout)
    fig1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    fig2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    
    ## Monthly Trend plot
    return [fig1,fig2]



##########################################################################################
############### main function
def hj_trend_chart(df):
    #### Description
    ## input : file name list
    ##         file must contain two columns whose name
    ##          date     : 일자
    ##          count : 일자별 count
    ## return : Null
    ## print two trend charts : Monthly, Weekly
    ## for 5 topic categories.
    #### Example
    ## file_name = ['filename1.xlsx','filename2.xlsx']
    ## hj_trend_chart(file_name)

    #### data prepocessing : data -> week_df, month_df
    # df = []
    # for i in file_names:
    #    df.append(pd.read_excel(i))

    month_df = []
    for i in df:
        month_df.append(make_month_num(i))

    week_df = []
    for i in df:
        week_df.append(make_week_num(i))


    ### Monthly trend plot
    month_scatter_plot = []
    for i in range(len(month_df)):
        month_scatter_plot.append(go.Scatter(
            x=list(month_df[i]['month_num']),
            y=list(month_df[i]['count']),
            name='Cate' + str(i),
            line=dict(width=1.5)))

    layout = go.Layout(
        font={
            "family": "뫼비우스 Regular",
            "size": 15
        },

        title={
         "text": '시간 흐름에 따른 관심도 변화, 월별',
            "font": {
                "size": 24,

            }
        }
        ,
        xaxis=dict(
            title='Date',
            titlefont=dict(
             family='뫼비우스 Regular',
             size=18,
             color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title='Count',
            titlefont=dict(
                family='뫼비우스 Regular',
                size=18,
                color='#7f7f7f'
            )
        ),
        hoverlabel={
            "font": {
                "family": "뫼비우스 Regular",

            }
        }
    )

    fig1 = go.Figure(data=month_scatter_plot, layout=layout)
    # ploting monthly trend chart

    ## Weekly Trend plot
    week_scatter_plot = []
    for i in range(len(week_df)):
        week_scatter_plot.append(go.Scatter(
            x=list(week_df[i]['week_num_new']),
            y=list(week_df[i]['count']),
            name='Cate' + str(i),
            line=dict(width=1.5)))


        layout = go.Layout(
            font={
                "family": "뫼비우스 Regular",
                "size": 15
            },

            title={
                "text": '시간 흐름에 따른 관심도 변화, 주별',
                "font": {
                    "size": 24,

                }
            },
            xaxis=dict(
                title='Date',
                titlefont=dict(
                    family='뫼비우스 Regular',
                    size=18,
                    color='#7f7f7f')
            ),
            yaxis=dict(
                title='Count',
                titlefont=dict(
                    family='뫼비우스 Regular',
                    size=18,
                    color='#7f7f7f')
            ),
            hoverlabel={
                "font": {
                    "family": "뫼비우스 Regular"}
            }
        )

    fig2 = go.Figure(data=week_scatter_plot, layout=layout)
    fig1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    fig2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    ## Monthly Trend plot
    return [fig1,fig2]


# In[6]:


def create_plot(df):

    layout = go.Layout(
        font={
            "family": "뫼비우스 Regular",
            "size": 15
        },

        title={
         "text": '시간 흐름에 따른 관심도 변화, 월별',
            "font": {
                "size": 24,

            }
        }
        ,
        xaxis=dict(
            title='Date',
            titlefont=dict(
             family='뫼비우스 Regular',
             size=18,
             color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title='Count',
            titlefont=dict(
                family='뫼비우스 Regular',
                size=18,
                color='#7f7f7f'
            )
        ),
        hoverlabel={
            "font": {
                "family": "뫼비우스 Regular",

            }
        }
    )
    
#     iplot([{"x": df['monthonly'], "y": df['counts']}])
    data = [go.Scatter(x=df['월'] ,y=df['count'])]
    fig = go.Figure(data=data, layout=layout)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
#     df = pd.read_csv('./stat_perday.csv')
#     data = [go.Scatter(x=df['dateonly'] ,y=df['counts'])]
#     graphJSON2 = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


# In[7]:


def plot_allgraph(idx):
    df1 = df[df.index.isin(assign_list[idx])]
#     df1['시간'] =df1['접수시각'].apply(lambda x: x[:2]) 
#     df1['월'] = df1['datetime'].apply(lambda x:x[:7])
    stat_time = df1.groupby(['시간']).size().reset_index(name='counts')
    stat_day = df1.groupby(['요일']).size().reset_index(name='counts')
    stat_month = df1.groupby(['월']).size().reset_index(name='counts')
    
    tra = go.Bar(x=["월","화","수","목","금","토","일"],y=stat_day['counts'],marker={'color':'rgba(26, 100, 219, 0.7)'})
    lay = go.Layout(font={"family":"뫼비우스 Regular","size":15},title="Topic %s의 요일 별 통계"%(idx+1),xaxis={"title":"요일"})
    fig = go.Figure(data=[tra], layout=lay)
    graph_day = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    tra = go.Bar(x=stat_time['시간'],y=stat_time['counts'],marker={'color':'rgba(26, 100, 219, 0.7)'})
    lay = go.Layout(font={"family":"뫼비우스 Regular","size":15},title="Topic %s의 시간 별 통계"%(idx+1),xaxis={"title":"시간"})
    fig = go.Figure(data=[tra], layout=lay)
    graph_time = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


    tra = go.Scatter(x=stat_month['월'],y=stat_month['counts'],marker={'color':'rgba(245,128,37, 0.7)'})
    lay = go.Layout(font={"family":"뫼비우스 Regular","size":15},title="Topic %s의 월 별 통계"%(idx+1),xaxis={"title":"날짜"})
    fig = go.Figure(data=[tra], layout=lay)
    graph_month = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    

    sent1 = sent[sent.index.isin(assign_list[idx])]
    donut = sent1.groupby(['sent_total']).size().reset_index(name="counts")
    tra = go.Pie(labels=donut['sent_total'], values=donut['counts'], hole=.7)
    lay = go.Layout(font={"family":"뫼비우스 Regular"},legend={"font":{"size":18}},annotations=[{"text":"감성분석","x":0.5,"y":0.5,"showarrow":False,"font":{"family":"뫼비우스 Regular","size":35,"color":"black"}}])
    fig = go.Figure(data=[tra], layout=lay)
    donut_plot =  json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return graph_day, graph_time, graph_month, donut_plot, len(df1)


# In[8]:


# directory = '2012-05-15_2019-03-03_3'
# topics = 3

def graph_topic_dir(directory, topics):
    len_list =[]
    day_list = []
    time_list = []
    month_list = []
    donut_list = []
# df = pd.read_excel('./dataset/syc_df.xlsx')
    assign_list_t = []
    for i in range(0,topics):
        a = pd.read_excel('./content/%s/TOPIC_XLSX/할당문서_%s.xlsx'%(directory,i))
        assign_list_t.append(list(a.index))
    
    key_list = []
    
    for i in range(0,topics):
        df1 = df[df.index.isin(assign_list_t[i])]
        len_list.append(len(df1))
        key_t = pd.read_excel('./content/%s/TOPIC_XLSX/단어빈도수_%s.xlsx'%(directory,i))[:5]

        stat_time = df1.groupby(['시간']).size().reset_index(name='counts')
        stat_day = df1.groupby(['요일']).size().reset_index(name='counts')
        stat_month = df1.groupby(['월']).size().reset_index(name='counts')

        tra = go.Bar(x=["월","화","수","목","금","토","일"],y=stat_day['counts'],marker={'color':'rgba(26, 100, 219, 0.7)'})
        lay = go.Layout(font={"family":"뫼비우스 Regular","size":15},title="Topic %s의 요일 별 통계"%(i+1),xaxis={"title":"요일"})
        fig = go.Figure(data=[tra], layout=lay)
        graph_day = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        tra = go.Bar(x=stat_time['시간'],y=stat_time['counts'],marker={'color':'rgba(26, 100, 219, 0.7)'})
        lay = go.Layout(font={"family":"뫼비우스 Regular","size":15},title="Topic %s의 시간 별 통계"%(i+1),xaxis={"title":"시간"})
        fig = go.Figure(data=[tra], layout=lay)
        graph_time = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        tra = go.Scatter(x=stat_month['월'],y=stat_month['counts'],marker={'color':'rgba(245,128,37, 0.7)'})
        lay = go.Layout(font={"family":"뫼비우스 Regular","size":15},title="Topic %s의 월 별 통계"%(i+1),xaxis={"title":"날짜"})
        fig = go.Figure(data=[tra], layout=lay)
        graph_month = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        sent1 = sent[sent.index.isin(assign_list_t[i])]
        donut = sent1.groupby(['sent_total']).size().reset_index(name="counts")
        tra = go.Pie(labels=donut['sent_total'], values=donut['counts'], hole=.7)
        lay = go.Layout(font={"family":"뫼비우스 Regular"},legend={"font":{"size":18}},annotations=[{"text":"감성분석","x":0.5,"y":0.5,"showarrow":False,"font":{"family":"뫼비우스 Regular","size":35,"color":"black"}}])
        fig = go.Figure(data=[tra], layout=lay)
        donut_plot =  json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        day_list.append(graph_day)
        time_list.append(graph_time)
        month_list.append(graph_month)
        donut_list.append(donut_plot)
        key_list.append(key_t.values)

    return day_list, time_list, month_list, key_list, donut_list, len_list


# # main flask module

# In[ ]:


##### from flask import Flask, render_template, request


app = Flask(__name__)

@app.route("/loadkeyword", methods=['GET','POST'])
def load_keyword():
    date = request.args.get('data')
    r1,r2,r3,r4,r5 = make_keyword(date)

    return render_template('give_data.html',r1=r1,r2=r2,r3=r3,r4=r4,r5=r5)
    
    

@app.route("/all_graph")
def allgraph():
    graph_list = []
    key_list = []
    do_list = []
    leng_list=[]
    for i in range(0,5):
        gr1,gr2, gr3, d1, leng = plot_allgraph(i)
        graph_list.append([gr1,gr2,gr3])
        key_list.append(key[i].values)
        do_list.append(d1)
        leng_list.append(leng)

    
    
    return render_template('tot_info.html',plot=graph_list, key=key_list, donut= do_list,
                          length=len(df),tot_len = leng_list)

@app.route("/graph")
def graph():

#     gr1,gr2 = create_plot()
    hj_graph3 = create_plot(hj_graph)
    hj_graph1, hj_graph2 = hj_trend_chart(hj_graph_list)
    return render_template('graph.html',plot=hj_graph1,plot2=hj_graph2,plot3=hj_graph3)

@app.route("/")
def index():
    
    return render_template("index.html")

@app.route("/index")
def index2():
    
    return render_template("index.html")

@app.route("/keyword")
def keyword():
    
    return render_template("keyword.html")

@app.route("/select")
def select():

    dir_li = []
    term_li = []
    to_num_li = []
    for directory in os.listdir('./content'):
        if not directory in ['LDA','TOPIC_PNG','TOPIC_XLSX']:
            info = directory.split('_')
            term_li.append("%s ~ %s"%(info[0],info[1]))
            to_num_li.append(info[-1])
            dir_li.append(directory)
    return render_template("select.html",dirs=dir_li,terms=term_li,tonums=to_num_li)

@app.route("/dash")
def dash():

    return render_template("dashboard.html")

@app.route("/analyze", methods=['GET','POST'])
def analyze():
    if request.method == 'POST':
        result = request.form
        start = result['from']
        end = result['to']
        num = result['topic_num']

        if start>end:
            return "입력 값이 이상합니다."
        if (not (start)) or (not(end)):
            return "날짜를 선택하세요."
        if not num:
            return "topic 수를 설정하세요."
#             return render_template("test.html",result = new_result)
        result = syc_tm(df,start,end,int(num))
        if result==1:
            return "분석 완료되었습니다."
        else:
            return "이미 분석된 값입니다."

@app.route("/show_graph", methods=['GET','POST'])
def show_graph():

    directory = request.args.get('dir')
    topics = request.args.get('top')
    
    info = directory.split('_')
    term = "%s ~ %s"%(info[0],info[1])
    
    day,time,month,key, donut, leng= graph_topic_dir(directory, int(topics))
    return render_template('dashboard.html',term= term,direc=directory, topics=topics, day=day,time=time,month=month,key=key,donut=donut, length=sum(leng), tot_len=leng)
    

if __name__ == "__main__":
    app.run(host='59.29.224.81',threaded=True)


# In[23]:


for i in range(0,5):
    df[df.index.isin(assign_list[i])].to_excel('./checkitout_%s.xlsx'%(i))

