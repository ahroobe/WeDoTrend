#### import packages.
import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import datetime

def make_month_num(df):
    ### return data which has month_number
    date_data = pd.to_datetime(df['date'])
    month_num = date_data.dt.strftime('%Y-%m')
    df['month_num'] = month_num
    df_sum_month = df.groupby(['month_num']).sum()
    df_sum_month['month_num']= df_sum_month.index
    df_sum_month.index = list(range(len(df_sum_month)))
    return df_sum_month


def make_week_num(df):
    ### return data which has week_number
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
    ## Monthly Trend plot
    return [fig1,fig2]