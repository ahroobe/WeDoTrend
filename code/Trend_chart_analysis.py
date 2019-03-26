import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go

# from DateTime import DateTime
# import datetime
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import datetime


##### date - counts data
##### plug in a file name with 5 categories data sets 
df = []
for i in range(5):
    df.append(pd.read_excel('cate_'+str(i)+'.xlsx'))
#print(df[0].head())
# df = pd.read_csv('stat_perday.csv')



def make_month_num(df):
    date_data = pd.to_datetime(df['date'])
    month_num = date_data.dt.strftime('%Y-%m')
    df['month_num'] = month_num
    df_sum_month = df.groupby(['month_num']).sum()
    df_sum_month['month_num']= df_sum_month.index
    df_sum_month.index = list(range(len(df_sum_month)))
    return df_sum_month


def make_week_num(df):
    date_data = pd.to_datetime(df['date'])
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


month_df = []
for i in df:
    month_df.append(make_month_num(i))

print(month_df[1].head())
week_df = []

for i in df:
    week_df.append(make_week_num(i))

print(week_df[1].head())



# plot_month_list = [{'x' : i['month_num'],'y':i['년도월일']} for i in month_df]

### Monthly trend plot

month_scatter_plot = []
for i in range(len(month_df)):
    month_scatter_plot.append(go.Scatter(
    x = list(month_df[i]['month_num']),
    y = list(month_df[i]['년도월일']),
    name = 'Cate'+ str(i)))


layout = go.Layout(
    font={
        "family":"뫼비우스 Regular",
        "size":15
    },

     title ={
         "text": '시간 흐름에 따른 관심도 변화, 월별',
         "font":{
             "size":24,
             
         }
     }
     ,
    xaxis=dict(
        title='Date',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Count',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    hoverlabel = {
        "font" : {
            "family":"뫼비우스 Regular",

        }
    }
)

fig = go.Figure(data=month_scatter_plot, layout=layout)
iplot(fig)






## Weekly Trend plot
week_scatter_plot = []
for i in range(len(week_df)):
    week_scatter_plot.append(go.Scatter(
    x = list(week_df[i]['week_num_new']),
    y = list(week_df[i]['년도월일']),
    name = 'Cate'+ str(i)))


layout = go.Layout(
    font={
        "family":"뫼비우스 Regular",
        "size":15
    },

     title ={
         "text": '시간 흐름에 따른 관심도 변화, 주별',
         "font":{
             "size":24,
             
         }
     }
     ,
    xaxis=dict(
        title='Date',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Count',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    hoverlabel = {
        "font" : {
            "family":"뫼비우스 Regular",
        }
    }
)

fig = go.Figure(data=week_scatter_plot, layout=layout)
iplot(fig)
