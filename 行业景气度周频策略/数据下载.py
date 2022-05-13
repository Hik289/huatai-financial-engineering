# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:46:35 2020

@author: wjxra
"""

import pandas as pd
import pymongo
import os

class data_model():
    
    # -------------------------------------------------------------------------
    # 加载数据库信息
    # -------------------------------------------------------------------------
    def __init__(self):
                
        # 获取MondoDB数据库链接
        self.client = pymongo.MongoClient("localhost", 27017)
        
        # 获取股票数据库对象
        self.stock_database = self.client["xquant_stock"]

        # 获取股票财报数据库对象
        self.stock_financial_database = self.client["xquant_stock_financial"]
        
        # 获取行业数据库对象
        self.indus_database = self.client["xquant_indus"]

        # 获取一致预取数据库对象
        self.est_database = self.client["xquant_est"]
        
        
    # -------------------------------------------------------------------------
    # 加载数据库信息
    # cursor      mongodb数据标签cursor
    # chunk_size  划分片数
    # -------------------------------------------------------------------------
    def cursor2dataframe(self, cursor, chunk_size: int):
        
        records = []  # 记录单片数据，写入dataframe
        frames = []   # 记录不同dataframe，拼接起来
        
        # 记录数据
        for i, record in enumerate(cursor):
            records.append(record)
            if i % chunk_size == chunk_size - 1:
                frames.append(pd.DataFrame(records))
                records = []
                
        # dataframe合并  
        if records:
                frames.append(pd.DataFrame(records))
                
        return pd.concat(frames)


    # -------------------------------------------------------------------------
    # 获取特定时间范围内的某个特定数据
    # database    数据库
    # collection  数据集（表）
    # start_date  开始时间
    # end_date    终止时间 
    # date_name   日期类型
    # stock_name  股票类型
    # target      调取数据目标
    # -------------------------------------------------------------------------
    def get_specific_data(self, database, collection, start_date, end_date,
                             date_name, stock_name, target):
      
        # 获取股票市值以及估值数据
        db_collection = database[collection]
        
        # 转换成pandas格式
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # 读取数据        
        if collection in ['AShareBalanceSheet','AShareCashFlow','AShareIncome']:
            cursor = db_collection.find({date_name:{
                                "$gte":start_date.strftime("%Y%m%d"),
                                "$lte":end_date.strftime("%Y%m%d")}, 'STATEMENT_TYPE':408001000},
                                {'_id':0, target:1, stock_name:1, date_name:1}).sort('$natural',1)
        else:
            
            cursor = db_collection.find({date_name:{
                                "$gte":start_date.strftime("%Y%m%d"),
                                "$lte":end_date.strftime("%Y%m%d")}},
                                {'_id':0, target:1, stock_name:1, date_name:1}).sort('$natural',1)
            
        # 读取数据, 存成DataFrame格式
        data = self.cursor2dataframe(cursor, 100000)
        
        # 业绩预告数据容易出现重复
        if collection == 'AShareProfitNotice':         
            
            # 相同股票数据去重
            data = data.sort_values(date_name)
            data.drop_duplicates(subset=[stock_name, date_name], keep='last', inplace=True) 
            
        # 重新整理数据index和columns
        data = data.pivot(index=date_name, columns=stock_name)[target]
        
        # index重新改写
        data.index = pd.to_datetime(data.index)
        
        return data
    
    
    # -------------------------------------------------------------------------
    # 获取特定时间范围内的所有数据：
    # database    数据库
    # collection  数据集（表）
    # start_date  开始时间
    # end_date    终止时间 
    # -------------------------------------------------------------------------
    def get_all_data(self, database, collection, start_date, end_date, date_name):
      
        # 获取股票市值以及估值数据
        db_collection = database[collection]
        
        # 转换成pandas格式
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # 读取数据        
        cursor = db_collection.find({date_name:{
                            "$gte":start_date.strftime("%Y%m%d"),
                            "$lte":end_date.strftime("%Y%m%d")}},
                            {'_id':0}).sort('$natural',1)

        # 读取数据, 存成DataFrame格式
        data = self.cursor2dataframe(cursor, 10000)
                
        return data
    
    
    # -------------------------------------------------------------------------
    # 获取特定时间范围内的所有数据：
    # database    数据库
    # collection  数据集（表）
    # start_date  开始时间
    # end_date    终止时间 
    # data_type   数据类型
    # -------------------------------------------------------------------------
    def get_all_data_with_type(self, database, collection, start_date, end_date, date_name, data_type):
      
        # 获取股票市值以及估值数据
        db_collection = database[collection]
        
        # 转换成pandas格式
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # 读取数据        
        cursor = db_collection.find({date_name:{
                            "$gte":start_date.strftime("%Y%m%d"),
                            "$lte":end_date.strftime("%Y%m%d")},
                            'ROLLING_TYPE': data_type},
                            {'_id':0}).sort('$natural',1)

        # 读取数据, 存成DataFrame格式
        data = self.cursor2dataframe(cursor, 10000)
                
        return data
    
    
    # -------------------------------------------------------------------------
    # 获取特定时间范围内的所有数据：
    # database    数据库
    # collection  数据集（表）
    # start_date  开始时间
    # end_date    终止时间 
    # data_type   数据类型
    # -------------------------------------------------------------------------
    def get_AShareConsensusData(self, database, collection, start_date, end_date, date_name, data_type):
      
        # 获取股票数据库
        db_collection = database[collection]
        
        # 转换成pandas格式
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # 读取数据        
        cursor = db_collection.find({date_name:{
                            "$gte":start_date.strftime("%Y%m%d"),
                            "$lte":end_date.strftime("%Y%m%d")},
                            'S_EST_YEARTYPE': data_type,
                            'CONSEN_DATA_CYCLE_TYP':263003000},
                            {'S_INFO_WINDCODE':1, 'EST_DT':1,
                             'S_EST_EPSINSTNUM':1, 'S_EST_ROEINSTNUM':1,
                             'S_EST_NETPROFITINSTNUM':1,
                             'S_EST_CPSINSTNUM':1,
                             'S_EST_DPSINSTNUM':1,
                             'S_EST_BPSINSTNUM':1,
                             'S_EST_EBITINSTNUM':1,
                             'S_EST_EBITDAINSTNUM':1,
                             'S_EST_EBTINSTNUM':1,
                             'S_EST_OPROFITINSTNUM':1,
                             'S_EST_MAINBUSINCINSTNUM':1,
                             '_id':0}).sort('$natural',1)

        # 读取数据, 存成DataFrame格式
        data = self.cursor2dataframe(cursor, 10000)
                
        return data


      
    
if __name__ == '__main__':
        
    model = data_model()
    
    start_date = '2005-01-01'
    end_date = '2021-05-31'
    quarterly_end_date = '2021-03-31'
    
# =============================================================================
#  读取股票列表
# =============================================================================
    
    # 最新A股信息
    stock_info = pd.DataFrame(list(model.stock_database["AShareDescription"].find({}, {'_id':0})))
    stock_info = stock_info.set_index('S_INFO_WINDCODE')
    
    # 剔除A和T开头的股票代码
    stock_info = stock_info.loc[[not i.startswith('A') for i in stock_info.index], :]
    stock_info = stock_info.loc[[not i.startswith('T') for i in stock_info.index], :]
    
    # 股票代码顺序重置
    stock_info.sort_index(inplace=True)
    stock_info.to_pickle('data/basic/stock_info')
    stock_info = pd.read_pickle('data/basic/stock_info')
    
    
# =============================================================================
#   读取映射关系
# =============================================================================
    
    # 中信行业信息
    indus_info = pd.read_excel('data/中信行业底层库代码名称.xlsx', index_col=0)
    
    # 华泰行业划分
    target_indus = indus_info[indus_info['板块'].notnull()]
    target_indus.to_pickle('data/basic/indus_info')    

    # 底层库行业代码替换成行业名称
    name_code_to_name = dict(zip(indus_info['行业代码'], indus_info['行业名称']))
    name_code_to_name['nan'] = 'nan'
    
    # Wind底层库行业代码替换成行业名称
    name_windcode_to_name = dict(zip(indus_info['Wind代码'], indus_info['行业名称']))
    name_windcode_to_name['nan'] = 'nan'
    
    
# =============================================================================
# 行业归属
# =============================================================================
    
    # 日频信息
    stock_daily_info = {}
    
    # 中信一级行业归属
    indus1_belong = model.get_specific_data(database = model.indus_database,
                                      collection = 'AShareIndustriesClassCITICS',
                                      start_date = start_date,
                                      end_date = end_date,
                                      date_name = 'date',
                                      stock_name = 'stock_code',
                                      target = 'cs_indus1_code') 

    # 中信二级行业归属
    indus2_belong = model.get_specific_data(database = model.indus_database,
                                      collection = 'AShareIndustriesClassCITICS',
                                      start_date = start_date,
                                      end_date = end_date,
                                      date_name = 'date',
                                      stock_name = 'stock_code',
                                      target = 'cs_indus2_code') 
    
    # 一二级行业混合
    indus_belong = indus1_belong.copy()
    
    # 替换食品饮料和非银行金融行业    
    indus1_belong = indus1_belong.astype('str')    
    indus2_belong = indus2_belong.astype('str')    
    indus_belong = indus_belong.astype('str')    
    indus_belong[indus1_belong=='b10j'] = indus2_belong[indus1_belong=='b10j']
    indus_belong[indus1_belong=='b10m'] = indus2_belong[indus1_belong=='b10m'] 
    
    # 底层库代码替换成行业名称
    indus_belong_name = indus_belong.applymap(lambda x: name_code_to_name[x])
    
    # 行业归属数据存储
    indus_belong_name = indus_belong_name.reindex(columns=stock_info.index) # 股票列表更新
    indus_belong_name.to_pickle('data/daily/stock/indus_belong')
    print('数据记录存储：indus_belong')


# =============================================================================
# 行业指数收盘价数据 - 用于策略回测
# =============================================================================

    # 按照报告期读取数据
    indus_close = model.get_specific_data(database = model.indus_database,
                                    collection = 'AIndexIndustriesEODCITICS',
                                    start_date = '2004-12-31',
                                    end_date = end_date,
                                    date_name='TRADE_DT',           # 按照交易日（TRADE_DT）读取数据
                                    stock_name='S_INFO_WINDCODE',   # 股票代码
                                    target='S_DQ_CLOSE')            # 读取收盘价数据
    
    # 新旧版映射关系
    map_list = [['CI005157.WI','CI005822.WI'],   # 其他饮料Ⅱ替换为饮料
                ['CI005158.WI','CI005823.WI'],   # 旧版食品替换为新版食品
                ['CI005167.WI','CI005828.WI']]   # 信托及其他替换为多元金融
    
    for index in map_list:        
        indus_close.loc[:'2019-11-29', index[1]] = indus_close.loc[:'2019-11-29', index[0]] / \
            indus_close.loc['2019-11-29', index[0]] * indus_close.loc['2019-11-29', index[1]] 
           
    indus_close = indus_close.reindex(columns=target_indus['Wind代码'])   

    # 行业Wind代码替换成名称
    indus_close.columns = list(map(lambda x: name_windcode_to_name[x], indus_close.columns))

    # 前向填充    
    indus_close = indus_close.bfill()
    
    # 数据存储
    indus_close = indus_close.reindex(columns=target_indus['行业名称'].tolist())
    indus_close.to_pickle('data/daily/indus/close')
    print('数据记录存储：indus_close')


# =============================================================================
# 个股市值数据
# =============================================================================
    
#    当日总市值	     S_VAL_MV
#    当日流通市值	 S_DQ_MV
#    当日总股本	     TOT_SHR_TODAY
#    当日流通股本	 FLOAT_A_SHR_TODAY


    # 获取股票市值以及估值数据
    db_collection = model.stock_database['AShareEODDerivativeIndicator']
    
    # 转换成pandas格式
    start_time = pd.to_datetime(start_date)
    end_time = pd.to_datetime(end_date)
    
    # 读取数据        
    cursor = db_collection.find({'TRADE_DT':{
                        "$gte":start_time.strftime("%Y%m%d"),
                        "$lte":end_time.strftime("%Y%m%d")}},
                        {'S_INFO_WINDCODE':1,'TRADE_DT':1,
                        'S_VAL_MV':1,'S_DQ_MV':1,
                        'TOT_SHR_TODAY':1,'FLOAT_A_SHR_TODAY':1,                             
                        '_id':0}).sort('$natural',1)

    # 读取数据, 存成DataFrame格式
    data = model.cursor2dataframe(cursor, 10000)
        
    # 数据存储
    data.to_pickle('data/backup/AShareEODDerivativeIndicator')
    data = pd.read_pickle('data/backup/AShareEODDerivativeIndicator')

    for data_name in ['S_VAL_MV', 'S_DQ_MV', 'TOT_SHR_TODAY', 'FLOAT_A_SHR_TODAY']:
        
        print(data_name)
        
        # 日期类型整理
        data['TRADE_DT'] = pd.to_datetime(data['TRADE_DT'], format='%Y-%m-%d')

        # 数据结构整理
        pivot = data.pivot_table(index='TRADE_DT', 
                                columns='S_INFO_WINDCODE',
                                values=data_name)
        pivot = pivot.fillna(method='ffill')
        pivot = pivot.reindex(columns=stock_info.index) # 股票列表更新
        
        # 数据存储
        pivot.to_pickle('data/daily/stock/{}'.format(data_name))
    
    
# =============================================================================
# 行业层面一致预期数据
# =============================================================================

    # 读取数据
    data = model.get_all_data(database = model.est_database,
                              collection = 'AIndexConsensusRollingData',
                              start_date = start_date,
                              end_date = end_date,
                              date_name = 'EST_DT')  # 按照报告期读取数据
    data.to_pickle('data/backup/AIndexConsensusRollingData')
    data = pd.read_pickle('data/backup/AIndexConsensusRollingData')
    
    # 数据类型
    types = ['FY1','FY2','FY3','FTTM','YOY','CAGR']
    for data_type in types:
        print(data_type)
        
        # 日期类型转换
        data['EST_DT'] =  pd.to_datetime(data['EST_DT'], format='%Y-%m-%d')
        
        # 净利润         	NET_PROFIT
        # 每股收益	        EST_EPS
        # 市盈率   	        EST_PE
        # PEG	            EST_PEG
        # 市净率	            EST_
        # 净资产收益率	    EST_ROE
        # 营业收入	        EST_OPER_REVENUE
        # 每股现金流	        EST_CFPS
        # 每股股利	        EST_DPS
        # 每股净资产	        EST_BPS
        # 息税前利润	        EST_EBIT
        # 息税折旧摊销前利润	EST_EBITDA
        # 利润总额	        EST_TOTAL_PROFIT
        # 营业利润	        EST_OPER_PROFIT
        # 营业成本及附加  	EST_OPER_COST

        # 给出wind一致预期数据映射关系
        factor_dict = {'EST_EPS_' + data_type: 'EST_EPS', 
                      'EST_PE_' + data_type: 'EST_PE',
                      'EST_PEG_' + data_type: 'EST_PEG', 
                      'EST_ROE_' + data_type: 'EST_ROE',
                      'EST_PB_' + data_type: 'EST_', 
                      'NET_PROFIT_' + data_type: 'NET_PROFIT',
                      'EST_OPER_REVENUE_' + data_type: 'EST_OPER_REVENUE',
                      'EST_CFPS_' + data_type: 'EST_CFPS',
                      'EST_DPS_' + data_type: 'EST_DPS', 
                      'EST_BPS_' + data_type: 'EST_BPS',
                      'EST_EBIT_' + data_type: 'EST_EBIT',
                      'EST_EBITDA_' + data_type: 'EST_EBITDA',
                      'EST_TOTAL_PROFIT_' + data_type: 'EST_TOTAL_PROFIT',
                      'EST_OPER_PROFIT_' + data_type: 'EST_OPER_PROFIT'}

        # 提取数据
        data1 = data[data.ROLLING_TYPE.isin([data_type])]
        
        # 数据类型整理以及存储
        for data_name, value in factor_dict.items():
            
            try: # 缺失数据需要跳过，特别对于CAGR会有缺失问题
                pivot = data1.pivot_table(index='EST_DT', 
                                              columns='S_INFO_WINDCODE',
                                              values=value)

                # 新旧版映射关系
                map_list = [['CI005157.WI','CI005822.WI'],   # 其他饮料Ⅱ替换为饮料
                            ['CI005158.WI','CI005823.WI'],   # 旧版食品替换为新版食品
                            ['CI005167.WI','CI005828.WI']]   # 信托及其他替换为多元金融
                
                for index in map_list:        
                    pivot.loc[:'2019-11-29', index[1]] = pivot.loc[:'2019-11-29', index[0]]
        
                # 数据整理
                pivot = pivot.loc[:, target_indus['Wind代码'].tolist()]                
                        
                # 用前期数据进行填充
                pivot = pivot.fillna(method='ffill')
                
                # 行业Wind代码替换成名称
                pivot.columns = list(map(lambda x: name_windcode_to_name[x], pivot.columns))
    
                # 数据存储
                pivot = pivot.reindex(columns=target_indus['行业名称'].tolist())
                pivot.to_pickle('data/daily/indus/{}'.format(data_name))
                
                print('数据记录存储：', data_type, data_name)
            except:
                print('数据存在缺失：', data_type, data_name)
    
    
# =============================================================================
# 个股层面一致预期数据
# =============================================================================
    
    # 数据类型
    types = ['FY1','FY2','FY3','FTTM','YOY','CAGR']
    
    # 按照报告期读取数据
    for data_type in types:
        data = model.get_all_data_with_type(database = model.est_database,
                                  collection = 'AShareConsensusRollingData',
                                  start_date = start_date,
                                  end_date = end_date,
                                  date_name = 'EST_DT',
                                  data_type = data_type) 

        data.to_pickle('data/backup/AShareConsensusRollingData_{}'.format(data_type))

    # 数据预处理
    for data_type in types:
        data = pd.read_pickle('data/backup/AShareConsensusRollingData_{}'.format(data_type))
        data['EST_DT'] =  pd.to_datetime(data['EST_DT'], format='%Y-%m-%d')
        
        # 数据类型
        factor_dict = {'EST_EPS_' + data_type: 'EST_EPS',
                      'EST_PE_' + data_type: 'EST_PE',
                      'EST_PEG_' + data_type: 'EST_PEG',
                      'EST_ROE_' + data_type: 'EST_ROE',
                      'EST_PB_' + data_type: 'EST_PB', 
                      'NET_PROFIT_' + data_type: 'NET_PROFIT',
                      'EST_OPER_REVENUE_' + data_type: 'EST_OPER_REVENUE',
                      'EST_CFPS_' + data_type: 'EST_CFPS',
                      'EST_DPS_' + data_type: 'EST_DPS',
                      'EST_BPS_' + data_type: 'EST_BPS',
                      'EST_EBIT_' + data_type: 'EST_EBIT',
                      'EST_EBITDA_' + data_type: 'EST_EBITDA',
                      'EST_TOTAL_PROFIT_' + data_type: 'EST_TOTAL_PROFIT',
                      'EST_OPER_PROFIT_' + data_type: 'EST_OPER_PROFIT'}
        
        # 数据整理
        data_cur = data[data.ROLLING_TYPE.isin([data_type])]
        for data_name, value in factor_dict.items():
            try:
                pivot = data_cur.pivot_table(index='EST_DT', columns='S_INFO_WINDCODE',
                                            values=value)
                
                # 用前期数据进行填充, 数据存储
                pivot = pivot.fillna(method='ffill')
                
                pivot = pivot.reindex(columns=stock_info.index) # 股票列表更新
                pivot.to_pickle('data/daily/stock/{}'.format(data_name))
                
                print('数据记录存储：', data_type, data_name)
            except:
                print('数据存在缺失：', data_type, data_name)
        
        
    


# =============================================================================
# 个股层面一致预期数据 - 预测机构家数
# =============================================================================
    
    type_list = ['FY1','FY2','FY3']
                
    # 读取数据
    # 每个类型分别读取
    for data_type in type_list:

        data = model.get_AShareConsensusData(database=model.est_database,
                                  collection='AShareConsensusData',
                                  start_date = start_date,
                                  end_date = end_date,
                                  date_name='EST_DT',
                                  data_type = data_type)

        data.to_pickle('data/backup/AShareConsensusData_{}'.format(data_type))

    for data_type in type_list:

        factor_dict = {'EST_EPS_' + data_type: 'S_EST_EPSINSTNUM',  
                      'EST_ROE_' + data_type: 'S_EST_ROEINSTNUM',
                      'EST_OPER_REVENUE_' + data_type: 'S_EST_MAINBUSINCINSTNUM',
                      'NET_PROFIT_' + data_type: 'S_EST_NETPROFITINSTNUM',
                      'EST_CFPS_' + data_type: 'S_EST_CPSINSTNUM',
                      'EST_DPS_' + data_type: 'S_EST_DPSINSTNUM', 
                      'EST_BPS_' + data_type: 'S_EST_BPSINSTNUM',
                      'EST_EBIT_' + data_type: 'S_EST_EBITINSTNUM',
                      'EST_EBITDA_' + data_type: 'S_EST_EBITDAINSTNUM',
                      'EST_TOTAL_PROFIT_' + data_type: 'S_EST_EBTINSTNUM',
                      'EST_OPER_PROFIT_' + data_type: 'S_EST_OPROFITINSTNUM'}


        data = pd.read_pickle('data/backup/AShareConsensusData_{}'.format(data_type))
        data['EST_DT'] =  pd.to_datetime(data['EST_DT'], format='%Y-%m-%d')

        for data_name, value in factor_dict.items():
            
            try:
                pivot = data.pivot_table(index='EST_DT', columns='S_INFO_WINDCODE',
                                        values=value)
                pivot = pivot.fillna(method='ffill')

                pivot = pivot.reindex(columns=stock_info.index) # 股票列表更新
                # 数据存储
                pivot.to_pickle('data/daily/stock/{}_num'.format(data_name))

                print('数据记录存储：', data_type, data_name)
            except:
                print('数据存在缺失：', data_type, data_name)
      


# =============================================================================
# 个股层面一致预期数据 - 业绩调整机构
# =============================================================================
    
    # 预测机构家数	NUM_EST_INST
    # 主营业务收入调高家数（与一个月前相比）	MAIN_BUS_INC_UPGRADE
    # 主营业务收入调低家数（与一个月前相比）	MAIN_BUS_INC_DOWNGRADE
    # 主营业务收入维持家数（与一个月前相比）	MAIN_BUS_INC_MAINTAIN
    # 净利润调高家数（与一个月前相比）	NET_PROFIT_UPGRADE
    # 净利润调低家数（与一个月前相比）	NET_PROFIT_DOWNGRADE
    # 净利润维持家数（与一个月前相比）	NET_PROFIT_MAINTAIN

    type_list = ['FY1','FY2','FY3']
    
    # 按照报告期读取数据
    for data_type in type_list:
        
        
        # 获取股票市值以及估值数据
        db_collection = model.est_database['AShareConsensusData']
        
        # 读取数据        
        cursor = db_collection.find({'EST_DT':{
                            "$gte":pd.to_datetime(start_date).strftime("%Y%m%d"),
                            "$lte":pd.to_datetime(end_date).strftime("%Y%m%d")},
                            'S_EST_YEARTYPE': data_type,
                            'CONSEN_DATA_CYCLE_TYP':263003000},
                            {'_id':0}).sort('$natural',1)

        # 读取数据, 存成DataFrame格式
        data = model.cursor2dataframe(cursor, 10000)

        data.to_pickle('data/backup/AShareConsensusData_{}'.format(data_type))
          
    type_list = ['FY2']
    for data_type in type_list:

        factor_dict = {'NUM_EST_INST_' + data_type: 'NUM_EST_INST',  
                      'MAIN_BUS_INC_UPGRADE_' + data_type: 'MAIN_BUS_INC_UPGRADE',
                      'MAIN_BUS_INC_DOWNGRADE_' + data_type: 'MAIN_BUS_INC_DOWNGRADE',
                      'MAIN_BUS_INC_MAINTAIN_' + data_type: 'MAIN_BUS_INC_MAINTAIN',
                      'NET_PROFIT_UPGRADE_' + data_type: 'NET_PROFIT_UPGRADE',
                      'NET_PROFIT_DOWNGRADE_' + data_type: 'NET_PROFIT_DOWNGRADE', 
                      'NET_PROFIT_MAINTAIN_' + data_type: 'NET_PROFIT_MAINTAIN'}

        data = pd.read_pickle('data/backup/AShareConsensusData_{}'.format(data_type))
        data['EST_DT'] =  pd.to_datetime(data['EST_DT'], format='%Y-%m-%d')

        for data_name, value in factor_dict.items():
            print(data_name, value)
            try:
                pivot = data.pivot_table(index='EST_DT', columns='S_INFO_WINDCODE', values=value)
                pivot = pivot.fillna(method='ffill')
                pivot = pivot.reindex(columns=stock_info.index) # 股票列表更新
                
                # 数据存储
                pivot.to_pickle('data/daily/stock/{}'.format(data_name))

                print('数据记录存储：', data_type, data_name)
            except:
                print('数据存在缺失：', data_type, data_name)
    
    
# =============================================================================
# 关注度类型的一致预期数据
# =============================================================================
    
    # 综合评级	S_WRATING_AVG
    # 评级机构数量	S_WRATING_INSTNUM
    # 调高家数（相比一月前）	S_WRATING_UPGRADE
    # 调低家数（相比一月前）	S_WRATING_DOWNGRADE
    # 维持家数（相比一月前）	S_WRATING_MAINTAIN
    # 买入家数	S_WRATING_NUMOFBUY
    # 增持家数	S_WRATING_NUMOFOUTPERFORM
    # 中性家数	S_WRATING_NUMOFHOLD
    # 减持家数	S_WRATING_NUMOFUNDERPERFORM
    # 卖出家数	S_WRATING_NUMOFSELL
    # 一致预测目标价	S_EST_PRICE
    # 目标价预测机构数	S_EST_PRICEINSTNUM
             
    # 读取数据
    data = model.get_all_data(database=model.est_database,
                              collection='AShareStockRatingConsus',
                              start_date = start_date,
                              end_date = end_date,
                              date_name='RATING_DT')

    data.to_pickle('data/backup/AShareStockRatingConsus')
    
    # 综合值周期类型S_WRATING_CYCLE取180天
    data = pd.read_pickle('data/backup/AShareStockRatingConsus')
    data = data[data['S_WRATING_CYCLE'] == 263003000]

    # 日期类型转换
    data['RATING_DT'] =  pd.to_datetime(data['RATING_DT'], format='%Y-%m-%d')

    data_list = ['S_WRATING_AVG', 'S_WRATING_INSTNUM','S_WRATING_UPGRADE', 
                  'S_WRATING_DOWNGRADE','S_WRATING_MAINTAIN','S_WRATING_NUMOFBUY',
                  'S_WRATING_NUMOFOUTPERFORM', 'S_WRATING_NUMOFHOLD',
                  'S_WRATING_NUMOFUNDERPERFORM', 'S_WRATING_NUMOFSELL',
                  'S_EST_PRICE', 'S_EST_PRICEINSTNUM']
                
    for data_name in data_list:
        
        # 列表展开，前值填充
        pivot = data.pivot_table(index='RATING_DT', columns='S_INFO_WINDCODE', values=data_name)
        pivot = pivot.fillna(method='ffill')
        
        # 股票列表更新
        pivot = pivot.reindex(columns=stock_info.index) 
        
        # 数据存储
        pivot.to_pickle('data/daily/stock/{}'.format(data_name))

        print('数据记录存储：', data_name)
        
    
# =============================================================================
#  # AShareBalanceSheet
# =============================================================================
    
# 公告日期	    ANN_DT
# 报告期	        REPORT_PERIOD
# 资产总计	    TOT_ASSETS
# 货币资金       MONETARY_CAP
# 应收账款       ACCT_RCV
# 应收票据       NOTES_RCV
# 交易性金融资产 TRADABLE_FIN_ASSETS
# 存货    INVENTORIES
# 固定资产       FIX_ASSETS
# 流动资产合计   TOT_CUR_ASSETS
# 短期借款  ST_BORROW
# 应付票据      NOTES_PAYABLE
# 应付账款      ACCT_PAYABLE
# 一年内到期的非流动负债 NON_CUR_LIAB_DUE_WITHIN_1Y
# 股东权益合计(不含少数股东权益)	TOT_SHRHLDR_EQY_EXCL_MIN_INT
# 负债合计	TOT_LIAB
# 流动负债合计	TOT_CUR_LIAB
    

    for data_name in ['TOT_ASSETS','MONETARY_CAP','ACCT_RCV','NOTES_RCV','TRADABLE_FIN_ASSETS',
                      'INVENTORIES','FIX_ASSETS','TOT_CUR_ASSETS','ST_BORROW','NOTES_PAYABLE','ACCT_PAYABLE',
                      'NON_CUR_LIAB_DUE_WITHIN_1Y','TOT_SHRHLDR_EQY_EXCL_MIN_INT','TOT_LIAB','TOT_CUR_LIAB']:
        print(data_name)
        data = model.get_specific_data(database=model.stock_financial_database,  # 读取股票财务数据库
                          collection='AShareBalanceSheet',  # 读取股票的资产负债表
                          start_date='1990-01-01',          # 读取数据（REPORT_PERIOD）的起始时间
                          end_date=end_date,            # 读取数据（REPORT_PERIOD）的终止时间
                          date_name='REPORT_PERIOD',        # 按照报告期（REPORT_PERIOD）读取数据
                          stock_name='S_INFO_WINDCODE',     # 股票代码
                          target=data_name)     # 读取非流动资产合计数据

        data = data.reindex(columns=stock_info.index) # 股票列表更新
        data.to_pickle('data/quarterly/stock/{}'.format(data_name))
        
        
# =============================================================================
#  中国A股现金流量表	AShareCashFlow
# =============================================================================
    
# 公告日期	    ANN_DT
# 报告期	        REPORT_PERIOD
# 销售商品、提供劳务收到的现金    CASH_RECP_SG_AND_RS
# 购建固定资产、无形资产和其他长期资产支付的现金   CASH_PAY_ACQ_CONST_FIOLTA
# 收回投资收到的现金     CASH_RECP_DISP_WITHDRWL_INVEST
# 取得投资收益收到的现金   CASH_RECP_RETURN_INVEST
# 投资活动产生的现金流量净额     NET_CASH_FLOWS_INV_ACT
# 吸收投资收到的现金     CASH_RECP_CAP_CONTRIB
# 筹资活动产生的现金流量净额     NET_CASH_FLOWS_FNC_ACT
# 固定资产折旧、油气资产折耗、生产性生物资产折旧   DEPR_FA_COGA_DPBA
# 无形资产摊销    AMORT_INTANG_ASSETS
# 长期待摊费用摊销  AMORT_LT_DEFERRED_EXP
# 期初现金及现金等价物余额  CASH_CASH_EQU_BEG_PERIOD
# 期末现金及现金等价物余额  CASH_CASH_EQU_END_PERIOD
# 现金及现金等价物净增加额    NET_INCR_CASH_CASH_EQU
# 经营活动产生的现金流量净额	NET_CASH_FLOWS_OPER_ACT
# 企业自由现金流量(FCFF)    FREE_CASH_FLOW
        


    for data_name in ['CASH_RECP_SG_AND_RS','CASH_PAY_ACQ_CONST_FIOLTA','CASH_RECP_DISP_WITHDRWL_INVEST',
                      'CASH_RECP_RETURN_INVEST','NET_CASH_FLOWS_INV_ACT','CASH_RECP_CAP_CONTRIB',
                      'NET_CASH_FLOWS_FNC_ACT','DEPR_FA_COGA_DPBA','AMORT_INTANG_ASSETS','AMORT_LT_DEFERRED_EXP',
                      'CASH_CASH_EQU_BEG_PERIOD','CASH_CASH_EQU_END_PERIOD','FREE_CASH_FLOW','NET_INCR_CASH_CASH_EQU']:
        print(data_name)
        data = model.get_specific_data(database=model.stock_financial_database,  # 读取股票财务数据库
                          collection='AShareCashFlow',  # 读取股票的资产负债表
                          start_date='1990-01-01',          # 读取数据（REPORT_PERIOD）的起始时间
                          end_date=end_date,            # 读取数据（REPORT_PERIOD）的终止时间
                          date_name='REPORT_PERIOD',        # 按照报告期（REPORT_PERIOD）读取数据
                          stock_name='S_INFO_WINDCODE',     # 股票代码
                          target=data_name)     # 读取非流动资产合计数据
        
        data = data.reindex(columns=stock_info.index) # 股票列表更新
        data.to_pickle('data/quarterly/stock/{}'.format(data_name))
      
        
# =============================================================================
# AShareIncome
# =============================================================================

# 公告日期	ANN_DT
# 报告期	    REPORT_PERIOD
# 营业总收入     TOT_OPER_REV
# 营业收入	OPER_REV
# 营业总成本  TOT_OPER_COST
# 营业成本	 LESS_OPER_COST
# 销售费用     LESS_SELLING_DIST_EXP
# 管理费用    LESS_GERL_ADMIN_EXP
# 财务费用     LESS_FIN_EXP
# 财务费用：利息费用    STMNOTE_FINEXP
# 研发费用     RD_EXPENSE
# 营业利润	OPER_PROFIT
# 利润总额	TOT_PROFIT
# 所得税   INC_TAX
# 应付普通股股利   COMSHARE_DVD_PAYABLE
# 净利润(含少数股东损益)	NET_PROFIT_INCL_MIN_INT_INC
# 净利润(不含少数股东损益)	NET_PROFIT_EXCL_MIN_INT_INC
# 息税前利润     EBIT
# 息税折旧摊销前利润     EBITDA

        
    for data_name in ['ANN_DT','OPER_REV', 'OPER_PROFIT', 'TOT_PROFIT', 'NET_PROFIT_EXCL_MIN_INT_INC',
                      'NET_PROFIT_INCL_MIN_INT_INC', 'TOT_OPER_COST', 'LESS_OPER_COST','TOT_OPER_REV',
                      'LESS_SELLING_DIST_EXP','LESS_GERL_ADMIN_EXP','LESS_FIN_EXP',
                      'STMNOTE_FINEXP','RD_EXPENSE','INC_TAX','COMSHARE_DVD_PAYABLE','EBIT','EBITDA']:
        print(data_name)
        data = model.get_specific_data(database=model.stock_financial_database,  # 读取股票财务数据库
                          collection='AShareIncome',  # 读取股票的资产负债表
                          start_date='1990-01-01',          # 读取数据（REPORT_PERIOD）的起始时间
                          end_date=end_date,            # 读取数据（REPORT_PERIOD）的终止时间
                          date_name='REPORT_PERIOD',        # 按照报告期（REPORT_PERIOD）读取数据
                          stock_name='S_INFO_WINDCODE',     # 股票代码
                          target=data_name)     # 读取非流动资产合计数据
        
        data = data.reindex(columns=stock_info.index) # 股票列表更新
        data.to_pickle('data/quarterly/stock/{}'.format(data_name))
    
    
# =============================================================================
#  AShareFinancialIndicator
# =============================================================================
    
# 同比增长率-基本每股收益(%)	S_FA_YOYEPS_BASIC
# 同比增长率-营业利润(%)	S_FA_YOYOP
# 同比增长率-利润总额(%)	S_FA_YOYEBT
# 同比增长率-归属母公司股东的净利润(%)	S_FA_YOYNETPROFIT
# 同比增长率-净资产收益率(摊薄)(%)	S_FA_YOYROE
# 营业收入同比增长率(%)	S_FA_YOY_OR
# 净资产收益率	S_FA_ROE
        
# 毛利    S_FA_GROSSMARGIN
# 经营活动净收益   S_FA_OPERATEINCOME
# 利息费用  S_STMNOTE_FINEXP
# 折旧与摊销     S_STM_IS
# 息税前利润     S_FA_EBIT
# 息税折旧摊销前利润     S_FA_EBITDA
# 企业自由现金流量(FCFF)    S_FA_FCFF
# 无息流动负债    S_FA_EXINTERESTDEBT_CURRENT
# 无息非流动负债   S_FA_EXINTERESTDEBT_NONCURRENT
# 全部投入资本    S_FA_INVESTCAPITAL
# 销售净利率     S_FA_NETPROFITMARGIN
# 销售毛利率     S_FA_GROSSPROFITMARGIN
# 销售成本率     S_FA_COGSTOSALES
# 销售期间费用率   S_FA_EXPENSETOSALES
# 销售费用/营业总收入    S_FA_SALEEXPENSETOGR
# 管理费用/营业总收     S_FA_ADMINEXPENSETOGR
# 财务费用/营业总收入    S_FA_FINAEXPENSETOGR
# 投入资本回报率   S_FA_ROIC
# 经营活动净收益/利润总额  S_FA_OPERATEINCOMETOEBT
# 经营活动产生的现金流量净额/营业收入    S_FA_OCFTOOR
# 资本支出/折旧和摊销    S_FA_CAPITALIZEDTODA
# 流动比率  S_FA_CURRENT
# 速动比率  S_FA_QUICK
# 经营活动产生的现金流量净额/流动负债    S_FA_OCFTOSHORTDEBT
# 产权比率  S_FA_DEBTTOEQUITY
# 经营活动产生的现金流量净额/负债合计    S_FA_OCFTODEBT
# 已获利息倍数(EBIT/利息费用)     S_FA_EBITTOINTEREST
# 存货周转率     S_FA_INVTURN
# 应收账款周转率   S_FA_ARTURN
# 流动资产周转率   S_FA_CATURN
# 固定资产周转率   S_FA_FATURN
# 总资产周转率    S_FA_ASSETSTURN
# 经营活动产生的现金流量净额／营业利润    S_FA_OCFTOPROFIT
# 货币资金／带息流动负债   S_FA_CASHTOLIQDEBTWITHINTEREST

            
    for data_name in ['S_FA_YOYEPS_BASIC','S_FA_YOYOP','S_FA_YOYEBT',
                      'S_FA_YOYNETPROFIT','S_FA_YOYROE',
                      'S_FA_YOY_OR','S_FA_ROE','S_FA_GROSSMARGIN',
                      'S_FA_OPERATEINCOME','S_STMNOTE_FINEXP',
                      'S_STM_IS','S_FA_EBIT','S_FA_EBITDA','S_FA_FCFF',
                      'S_FA_EXINTERESTDEBT_CURRENT','S_FA_EXINTERESTDEBT_NONCURRENT',
                      'S_FA_INVESTCAPITAL','S_FA_NETPROFITMARGIN','S_FA_GROSSPROFITMARGIN',
                      'S_FA_COGSTOSALES','S_FA_EXPENSETOSALES','S_FA_SALEEXPENSETOGR',
                      'S_FA_ADMINEXPENSETOGR','S_FA_FINAEXPENSETOGR','S_FA_ROIC',
                      'S_FA_OPERATEINCOMETOEBT','S_FA_OCFTOOR','S_FA_CAPITALIZEDTODA',
                      'S_FA_CURRENT','S_FA_QUICK','S_FA_OCFTOSHORTDEBT','S_FA_DEBTTOEQUITY',
                      'S_FA_OCFTODEBT','S_FA_EBITTOINTEREST','S_FA_INVTURN',
                      'S_FA_ARTURN','S_FA_CATURN','S_FA_FATURN','S_FA_ASSETSTURN',
                      'S_FA_OCFTOPROFIT','S_FA_CASHTOLIQDEBTWITHINTEREST',
                      'S_FA_YOY_OR','S_FA_ROE']:
        print(data_name)
        data = model.get_specific_data(database=model.stock_financial_database,  # 读取股票财务数据库
                          collection='AShareFinancialIndicator',  # 读取股票的资产负债表
                          start_date='1990-01-01',          # 读取数据（REPORT_PERIOD）的起始时间
                          end_date=end_date,            # 读取数据（REPORT_PERIOD）的终止时间
                          date_name='REPORT_PERIOD',        # 按照报告期（REPORT_PERIOD）读取数据
                          stock_name='S_INFO_WINDCODE',     # 股票代码
                          target=data_name)     # 读取非流动资产合计数据
        
        data = data.reindex(columns=stock_info.index) # 股票列表更新
        data.to_pickle('data/quarterly/stock/{}'.format(data_name))
        
        if data_name == 'S_FA_ROE':
            data_yoy = data - data.shift(4)
            data_yoy = data_yoy.reindex(columns=stock_info.index) # 股票列表更新
            data_yoy.to_pickle('data/quarterly/stock/S_FA_ROE_YOY')
        
  
# =============================================================================
#  # 读取业绩快报数据
# =============================================================================
    
# 公告日期	    ANN_DT
# 报告期	        REPORT_PERIOD
# 营业收入(元)	OPER_REV
# 营业利润(元)	OPER_PROFIT
# 利润总额(元)	TOT_PROFIT
# 净利润(元)    	NET_PROFIT_EXCL_MIN_INT_INC
# 总资产(元) 	TOT_ASSETS
# 股东权益合计(不含少数股东权益)(元)	 TOT_SHRHLDR_EQY_EXCL_MIN_INT

# 同比增长率:营业收入	S_FA_YOYSALES
# 同比增长率:营业利润	S_FA_YOYOP
# 同比增长率:利润总额	S_FA_YOYEBT
# 同比增长率:归属母公司股东的净利润	S_FA_YOYNETPROFIT_DEDUCTED
# 同比增长率:基本每股收益	S_FA_YOYEPS_BASIC
# 同比增减:加权平均净资产收益率	S_FA_ROE_YEARLY

    for data_name in ['ANN_DT','S_FA_YOYSALES','S_FA_YOYOP','S_FA_YOYEBT',
                      'S_FA_YOYNETPROFIT_DEDUCTED','S_FA_YOYEPS_BASIC','S_FA_ROE_YEARLY']:
        print(data_name)
        data = model.get_specific_data(database=model.stock_financial_database,  # 读取股票财务数据库
                          collection='AShareProfitExpress',  # 读取股票的资产负债表
                          start_date='1990-01-01',          # 读取数据（REPORT_PERIOD）的起始时间
                          end_date=quarterly_end_date,            # 读取数据（REPORT_PERIOD）的终止时间
                          date_name='REPORT_PERIOD',        # 按照报告期（REPORT_PERIOD）读取数据
                          stock_name='S_INFO_WINDCODE',     # 股票代码
                          target=data_name)     # 读取非流动资产合计数据
        
        data = data.reindex(columns=stock_info.index) # 股票列表更新
        data.to_pickle('data/quarterly/stock/{}_exp'.format(data_name))


# =============================================================================
#  # 读取业绩预告数据
# =============================================================================
    
# 公告日期	S_PROFITNOTICE_DATE
# 预告净利润下限（万元）	S_PROFITNOTICE_NETPROFITMIN
# 预告净利润上限（万元）	S_PROFITNOTICE_NETPROFITMAX
# 预告净利润变动幅度下限（%）	S_PROFITNOTICE_CHANGEMIN
# 预告净利润变动幅度上限（%）	S_PROFITNOTICE_CHANGEMAX

    for data_name in ['S_PROFITNOTICE_DATE','S_PROFITNOTICE_NETPROFITMIN','S_PROFITNOTICE_NETPROFITMAX',
                      'S_PROFITNOTICE_CHANGEMIN','S_PROFITNOTICE_CHANGEMAX']:
        print(data_name)
        data = model.get_specific_data(database=model.stock_financial_database,  # 读取股票财务数据库
                          collection='AShareProfitNotice',  # 读取股票的资产负债表
                          start_date='2000-01-01',          # 读取数据（REPORT_PERIOD）的起始时间
                          end_date=quarterly_end_date,            # 读取数据（REPORT_PERIOD）的终止时间
                          date_name='S_PROFITNOTICE_PERIOD',        # 按照报告期（REPORT_PERIOD）读取数据
                          stock_name='S_INFO_WINDCODE',     # 股票代码
                          target=data_name)     # 读取非流动资产合计数据

        data = data.reindex(columns=stock_info.index) # 股票列表更新
        data.to_pickle('data/quarterly/stock/{}'.format(data_name))
  
    
    
    
    
