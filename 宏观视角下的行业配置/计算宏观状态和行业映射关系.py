# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 20:35:46 2021


"""

import pandas as pd
import numpy as np

indus_close = pd.read_pickle('data/close')

indus_info = pd.read_pickle('data/indus_info')

predict_idea = pd.read_excel('data/滚动因子观点.xlsx', index_col=0).resample('M').last()

seperate_period = pd.read_excel('data/因子周期划分结果.xlsx', index_col=0).resample('M').last()

indus_return = indus_close.resample('M').last().pct_change()

indus_return = indus_return.add(-indus_return.mean(axis=1), axis=0)

indus_return = indus_return.reindex(index=seperate_period.index)


data_up = pd.DataFrame(index=indus_info['行业名称'], 
                  columns=['增长上行', '通胀上行', '信用上行', '货币上行'])

data_down = pd.DataFrame(index=indus_info['行业名称'], 
                  columns=['增长下行', '通胀下行', '信用下行', '货币下行'])

for index in range(0, len(seperate_period.columns)):
    
    period = seperate_period.columns[index]
    
    data_period = seperate_period.loc[:, period]
    
    indus_return_summary_up = indus_return.loc[data_period==1,:].mean()
    
    data_up.iloc[:, index] = indus_return_summary_up - indus_return_summary_up.mean()
    
    indus_return_summary_down = indus_return.loc[data_period==-1,:].mean()
    
    data_down.iloc[:, index] = indus_return_summary_down - indus_return_summary_down.mean()
    
    
# 计算映射关系
factor_up = data_up.copy()
factor_up[data_up>0] = 1
factor_up[data_up<0] = -1
factor_up[(data_up<0.001) & (data_up>-0.001)] = 0

factor_down = data_down .copy()
factor_down[data_down>0] = 1
factor_down[data_down<0] = -1
factor_down[(data_down<0.001) & (data_down>-0.001)] = 0

for index in factor_up.index:
    for i in range(0,4):
        
        if (factor_up.loc[index, :].iloc[i] == 1) & (factor_down.loc[index, :].iloc[i] == 1):
            if data_up.loc[index, :].iloc[i] > data_down.loc[index, :].iloc[i]:
                factor_up.loc[index, :].iloc[i] = 1
                factor_down.loc[index, :].iloc[i] = 0
            else:
                factor_up.loc[index, :].iloc[i] = 0
                factor_down.loc[index, :].iloc[i] = 1
                
        if (factor_up.loc[index, :].iloc[i] == -1) & (factor_down.loc[index, :].iloc[i] == -1):
            if data_up.loc[index, :].iloc[i] > data_down.loc[index, :].iloc[i]:
                factor_up.loc[index, :].iloc[i] = 1
                factor_down.loc[index, :].iloc[i] = 0
            else:
                factor_up.loc[index, :].iloc[i] = 0
                factor_down.loc[index, :].iloc[i] = 1
                
# 最终结果
map_data = pd.concat([data_up,data_down], axis=1)          
map_factor = pd.concat([factor_up,factor_down], axis=1)
    
    
    