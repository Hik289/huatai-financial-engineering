# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 11:41:33 2018


"""

import pandas as pd
import numpy as np
import os

# -----------------------------------------------------------------------------
# 获取文件所在绝对路径
# -----------------------------------------------------------------------------
def get_file_path():
    
    return os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# -----------------------------------------------------------------------------
# 获取基本信息，读取路径 'data/basic/prop_name'
# -----------------------------------------------------------------------------
def get_basic_info(prop_name):
    
    file_path = get_file_path()

    return pd.read_pickle(file_path + '\\data\\basic\\{0}'.format(prop_name))


# -----------------------------------------------------------------------------
# 获取指定日频数据(读取路径 'data/daily/prpp_type/prop_name')，并做适配性修改
# [输入]
# prpp_type: 指标类型，可以是行业、股票等，会在同名文件路径下寻找
# prop_name：指标名称
# adj_param：数据调整参数，可以对读取的原始数据进行修正，可选参数包含
#            -- start_date:    数据截断的起始日期
#            -- end_date:      数据截断的终止日期
#            -- resample_date: 重采样日期序列
#            -- ipo_date:      上市日期，根据上市日期滤除无效数据
#            -- delist_date:   退市日期，根据退市日期过滤无效数据
# [输出]
# 调整后的DataFrame
# -----------------------------------------------------------------------------
def get_daily_info(prpp_type, prop_name, **adj_param):
    
    # 获取文件所在路径
    file_path = get_file_path()
    
    # 读取原始数据
    df = pd.read_pickle(file_path+'\\data\\daily\\{0}\\{1}'.format(prpp_type, prop_name))
    
    # 按上市日期置位无效数据
    if 'ipo_date' in adj_param:
        ipo_date = adj_param['ipo_date']
        df = df.apply(lambda x: np.where(x.index < ipo_date[x.name], np.nan, x))
    
    # 按退市日期置位无效数据
    if 'delist_date' in adj_param:
        delist_date = adj_param['delist_date']
        df = df.apply(lambda x: np.where(x.index > delist_date[x.name], np.nan, x))
    
    # 按重采样日期筛选数据
    if 'resample_date' in adj_param:
        df = df.loc[adj_param['resample_date'], :]
        
    # 按照截断日期剔除目标区间以前的数据
    if 'start_date' in adj_param:  
        df = df.loc[adj_param['start_date']:, :]
    
    # 按照截断日期剔除目标区间以后的数据
    if 'end_date' in adj_param:  
        df = df.loc[:adj_param['end_date'], :]
        
    return df

# -----------------------------------------------------------------------------
# 获取指定财报指标(读取路径 'data/quarterly/prpp_type/prop_name')，并做适配性修改
# [输入]
# prpp_type: 指标类型，可以是行业、股票等，会在同名文件路径下寻找
# prop_name：指标名称
# adj_param：数据调整参数，可以对读取的原始数据进行修正，可选参数包含
#            -- start_date:  数据截断的起始日期
#            -- end_date:    数据截断的终止日期
#            -- ipo_date:    上市日期，根据上市日期滤除无效数据
#            -- delist_date: 退市日期，根据退市日期过滤无效数据
#            -- transform：  类型转换，ttm（一年滚动平均），qfa（单季度）
# [输出]
# 调整后的DataFrame
# -----------------------------------------------------------------------------
def get_quarterly_info(prpp_type, prop_name, **adj_param):
    
    # 获取文件所在路径
    file_path = get_file_path()
    
    # 读取原始数据
    df = pd.read_pickle(file_path+'\\data\\quarterly\\{0}\\{1}'.format(prpp_type, prop_name))
    
    # 按上市日期置位无效数据
    if 'ipo_date' in adj_param:
        ipo_date = adj_param['ipo_date']
        df = df.apply(lambda x: np.where(x.index < ipo_date[x.name], np.nan, x))
    
    # 按退市日期置位无效数据
    if 'delist_date' in adj_param:
        delist_date = adj_param['delist_date']
        df = df.apply(lambda x: np.where(x.index > delist_date[x.name], np.nan, x))
    
    # 按类型转换要求修改数据（主要是原始利润表数据是累计值，可以转换为TTM或单季度）
    if 'transform' in adj_param:  
        transform_map = {'ttm' : transform_ttm, 'qfa' : transform_qfa}
        df = transform_map[adj_param['transform']](df)
    
    # 按照截断日期剔除目标区间以前的数据
    if 'start_date' in adj_param:  
        df = df.loc[adj_param['start_date']:, :]
    
    # 按照截断日期剔除目标区间以后的数据
    if 'end_date' in adj_param:  
        df = df.loc[:adj_param['end_date'], :]
        
    return df   


# -----------------------------------------------------------------------------
# 将原始利润表数据转换为TTM数据，注意这里要求输入数据为季频索引
# -----------------------------------------------------------------------------
def transform_ttm(df_factor):
    
    # 获取基本数据
    n_quarters = df_factor.shape[0]
    quarterly_dates = df_factor.index
        
    # 初始化TTM数据
    factor_ttm = np.nan * np.zeros_like(df_factor)
    
    # 从第5条数据开始遍历
    for row_index in range(4, n_quarters, 1):
        
        # 获取当前季报索引，并执行不同操作
        q_index = int(quarterly_dates[row_index].month/3) % 4
        
        if q_index == 0:
            # 如果是年报值，直接设置
            factor_ttm[row_index,:] = df_factor.iloc[row_index,:]
        else:
            # 如果是季报值，当前数据 + 去年年报 - 去年同期
            factor_ttm[row_index,:] = df_factor.iloc[row_index-q_index,:] + \
                    df_factor.iloc[row_index,:] - df_factor.iloc[row_index-4,:]
    
    # 生成结果
    df_ret = pd.DataFrame(factor_ttm, index=df_factor.index, columns=df_factor.columns)  
    
    # 去除前面的空值
    df_ret = df_ret.iloc[4:, :]
        
    return df_ret
    

# -----------------------------------------------------------------------------
# 将原始利润表数据转换为单季度数据，注意这里要求输入数据为季频索引
# -----------------------------------------------------------------------------
def transform_qfa(df_factor):

    # 获取基本数据
    n_quarters = df_factor.shape[0]
    quarterly_dates = df_factor.index
    
    # 从第一个一季度开始计算
    begin_index = list(quarterly_dates).index(
            next(x for x in quarterly_dates if x.month == 3))
        
    # 初始化单季度数据
    factor_qfa = np.nan * np.zeros_like(df_factor)
    
    # 从第一个一季度数据开始遍历
    for row_index in range(begin_index, n_quarters, 1):
        
        # 获取当前季报索引，并执行不同操作
        q_index = int(quarterly_dates[row_index].month/3) % 4
        
        if q_index == 1:
            # 如果是一季报，直接设置
            factor_qfa[row_index,:] = df_factor.iloc[row_index,:]
        else:
            # 如果是其他值，当前数据 - 上一期数据
            factor_qfa[row_index,:] = \
                    df_factor.iloc[row_index,:] - df_factor.iloc[row_index-1,:]
    
    # 生成结果
    df_ret = pd.DataFrame(factor_qfa, index=df_factor.index, columns=df_factor.columns)  
    
    # 去除前面的空值
    df_ret = df_ret.iloc[begin_index:, :]
        
    return df_ret



if __name__ == '__main__':
    
    # 获取所有股票的信息
    stock_info = get_basic_info('stock_info')
    
    # 获取一级行业基本信息
    indus_info = get_basic_info('indus_info')
    
    # 获取所有股票的营收数据，不做任何处理
    oper_rev1 = get_quarterly_info('stock', 'oper_rev')
  
    # 获取一级行业收盘价
    indus_close = get_daily_info('indus', 'close')

    # 获取一级行业收盘价，仅截取10年以后的数据
    indus_close = get_daily_info('indus', 'close', start_date='2010-01-01')


