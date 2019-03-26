# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:17:24 2019

@author: HPE
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

# Set Korean Font
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

# Copy and reset index
df = df_n.copy()
df = df.reset_index(drop=True)

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

# New column
# 접수일자, 요일, 접수시각, 년도, 월, 일

# Get data for a week
from datetime import datetime
from datetime import timedelta          

df['년도월일'] = [i[0:10] for i in df['datetime']]     

def convert_d(std_d, n):
    convert_date = datetime.strptime(std_d, "%Y-%m-%d").date()
    start_d = convert_date
    end_d = start_d + timedelta(days = -(n))
    return start_d, end_d
    

# Set today
standard = df['년도월일'][0]

lst = []
for i in range(len(df)):
    lst.append(df.iloc[i])
    start, end = convert_d(standard, 7)
    if df['년도월일'][i] == str(end):
        break

week_dat = pd.DataFrame(np.vstack([j for j in lst]))
week_dat.columns = df.columns
