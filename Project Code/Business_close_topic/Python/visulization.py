# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 22:00:13 2023

@author: Potatopie
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import os

plt.switch_backend('Agg')

outputfile_name = 'img_save'
input_file_name = "Businesses_Size.xlsx"
input_sheet_name = "Data"

strat_col = 3
num_feature = 6

start_line = 6
num_items = 85

num_slot = 5

num_gap = 14

list_num = [0,2,4,6,8,10]

data = np.zeros([num_slot,num_items,num_feature])

df = pd.read_excel(input_file_name, engine = "openpyxl",sheet_name = input_sheet_name)
industry_list = df.iloc[start_line:start_line + num_items,strat_col - 1].values
size_list = df.iloc[start_line - 2,strat_col:strat_col + num_feature].values

for idx in range(num_slot):
    if idx == 0:
        tmp_mt = df.iloc[start_line:start_line + num_items,strat_col:strat_col + num_feature].values
        data[idx,:,:] = tmp_mt
        strat_col = strat_col + 7
    else:
        tmp_mt = df.iloc[start_line:start_line + num_items,strat_col:strat_col + num_feature*2].values
        data[idx,:,:] = tmp_mt[:,list_num]
        strat_col = strat_col + num_gap
        
if not os.path.isdir(outputfile_name):
    os.mkdir(outputfile_name)

start_year = 2018
end_year = 2022
star_imp_date = '2020/06'
end_imp_date = '2020/10'
date_list = range(start_year,end_year + 1)

time_list = []

for year in date_list:
    year_str = str(year) + '/06'
    time_list.append(pd.to_datetime(year_str))

star_imp_date = pd.to_datetime(star_imp_date)
end_imp_date = pd.to_datetime(end_imp_date)

cal = 0
print('start generating!')
for line,industry in enumerate(industry_list):
    for col,size in enumerate(size_list):
        
        cal = cal + 1
        
        now_data = data[:,line,col]
        now_data = np.squeeze(now_data)


        fig = plt.figure(figsize=(16,9))

        ax = plt.subplot(111)

        ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m'))

        plt.xticks(pd.date_range(time_list[0],time_list[-1],freq='6M'),rotation=45)

        ax.plot(time_list,now_data,color='r')

        title = "The businesses with an annual turnover of (" + size + ") from the " + industry + " industry"
        if '/' in title:
            title = title.replace('/', ' ')
        ax.set_title(title,fontsize=12)
        
        ax.set_ylim(bottom=0.)

        ax.set_xlabel('Date of survey',fontsize=20)
        ax.set_ylabel('Number of businesses exist in 2018',fontsize=20,rotation=270)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        
        #ax.vlines([star_imp_date, end_imp_date], min(now_data), max(now_data), linestyles='dashed', colors='red')
        plt.fill_between([star_imp_date, end_imp_date], 0, max(now_data), facecolor='blue', alpha=0.2)
        
        plt.text(star_imp_date,max(now_data)*0.5,"Support Given",fontdict={'size':'10','color':'r'})
        
        plt.savefig(outputfile_name + '/' + title + '.png')
        
        plt.close()
        
        if cal%50 == 0:
            print(str(cal) + ' images have been generated, continue......')















