# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 20:15:46 2021

@author: kipcrk290
"""
import numpy as np
import pandas as pd
import pymongo

class data_download:
    
    def __init__(self):
        
        # 连接MongoDB数据库
        client = pymongo.MongoClient(host="localhost",port=27017)
        
        # 股票数据库
        self.stock_database = client["xquant_stock"]
        
        # 行业数据库
        self.indus_database = client["xquant_indus"]
        
        # 指数数据库
        self.index_database = client["xquant_index"]
        
        # 股票财报数据库
        self.stock_financial_database = client["xquant_stock_financial"]
        
        # 基金数据库
        self.fund_database = client["xquant_fund"]
        
        # 一致预期数据库
        self.est_database = client["xquant_est"]
    
    # ================================================================================
    # 把cursor转换为dataframe
    # ================================================================================
    def cursor2dataframe(self,cursor,chunk_size:int):
        
        # 记录单片数据后进行拼接
        records = []
        frames = []
        
        # 记录数据
        for i,record in enumerate(cursor):
            records.append(record)
            if i % chunk_size == chunk_size-1:
                frames.append(pd.DataFrame(records))
                records = []
        if records:
            frames.append(pd.DataFrame(records))
        
        # 数据拼接
        df = pd.concat(frames)
        
        return df
    
    # ================================================================================
    # 获取特定时间范围内的一些特定数据
    # database           数据库
    # collection         数据集（表）
    # start_date         开始日期
    # end_date           结束日期
    # date_name          日期数据标签
    # stock_name         股票数据标签
    # target_list        目标数据指标列表
    # stock_code         选择特定股票代码
    # conditions         补充数据调用条件
    # ================================================================================
    def get_specific_data(self,database,collection,start_date,end_date,date_name,
                               stock_name,target_list,stock_code=None,conditions=None):
        
        # 转换日期格式
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # 数据库调用条件
        query = {date_name:{"$gte":start_date.strftime("%Y%m%d"),
                            "$lte":end_date.strftime("%Y%m%d")}}
        if stock_code is not None:
            query.update({stock_name:{"$in":stock_code}})
        if conditions is not None:
            query.update(conditions)
        
        # 目标数据指标列表
        projection = {"_id":0,date_name:1,stock_name:1}
        for target in target_list:
            projection[target] = 1
        
        # 从数据库读取数据
        data = self.export_data(database,collection,query,projection)
        
        # 数据去重
        data = self.remove_duplicate(data,collection,date_name,stock_name)
        
        # 转换为dataframe格式 
        data_list = self.dataframe_pivot(data,date_name,stock_name,target_list)
        
        return data_list
    
    # ================================================================================
    # 获取特定时间范围内的所有数据
    # database           数据库
    # collection         数据集（表）
    # start_date         开始日期
    # end_date           结束日期
    # date_name          日期数据标签
    # ================================================================================
    def get_all_data(self,database,collection,start_date,end_date,date_name):
        
        # 转换日期格式
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # 数据库调用条件
        query = {date_name:{"$gte":start_date.strftime("%Y%m%d"),
                            "$lte":end_date.strftime("%Y%m%d")}}
        
        # 目标数据指标列表
        projection = {"_id":0}
        
        # 读取数据
        data = self.export_data(database,collection,query,projection)
        
        return data
    
    # ================================================================================
    # 从数据库读取数据
    # database           数据库
    # collection         数据集（表）
    # query              数据库调用条件
    # projection         目标数据指标列表
    # ================================================================================
    def export_data(self,database,collection,query,projection):
        
        # 读取数据集
        db_col = database[collection]
        
        # 读取数据，保存为dataframe格式
        cursor = db_col.find(query,projection).sort("$natural",1)
        data = self.cursor2dataframe(cursor,100000)
        
        return data
    
    # ================================================================================
    # 把合并的dataframe拆分为多个dataframe
    # conditions         补充数据调用条件
    # df                 目标dataframe
    # target_idx         行索引
    # target_col         列索引
    # target_list        目标数据指标列表
    # ================================================================================
    def dataframe_pivot(self,df,target_idx,target_col,target_list):
        
        df_list = []
        for name in target_list:
            
            # 重新整理index和columns
            temp_data = df.pivot(index=target_idx,columns=target_col,values=name)
            
            # 转换日期序列格式
            temp_data.index = pd.to_datetime(temp_data.index)
            
            # 检验数据是否全为nan
            if temp_data.notnull().values.sum() > 0:
                df_list.append(temp_data)
            else:
                df_list.append(None)
        
        return df_list
    
    # ================================================================================
    # 去掉转换后的dataframe中的重复行
    # ================================================================================
    def remove_duplicate(self,df,collection,date_name,stock_name,method="last"):
        
        # 部分数据集的数据可能出现重复，需要去掉重复数据
        if collection in ['AShareProfitNotice','HKIndexEODPrices','HKStockHSIndustriesMembers',
                          "AShareMoneyFlow","AShareEODDerivativeIndicator","AShareStockRatingConsus",
                          "AShareConsensusData","AShareBalanceSheet","AShareCashFlow","AShareIncome"]:
            
            # 相同股票数据去重
            df = df.sort_values(date_name)
            df_unique = df.drop_duplicates(subset=[stock_name,date_name],keep=method)
        else:
            df_unique = df
        
        return df_unique
    

if __name__ == "__main__":
    
    model = data_download()
    start_date = "1990-01-01"
    end_date = "2021-12-31"
    
# ================================================================================
# 股票基本信息
# ================================================================================

    print("股票基本信息")
    data = model.get_all_data(database=model.stock_database,
                              collection="AShareDescription",
                              start_date="2000-01-01",
                              end_date="2030-01-01",
                              date_name="OPDATE").set_index("S_INFO_WINDCODE")
    data = data[["S_INFO_NAME","S_INFO_EXCHMARKET","S_INFO_LISTDATE"]]
    
    # 剔除A和T开头的股票代码
    keep_index = list(map(lambda i:data.index[i][0] not in ["A","T"],range(data.shape[0])))
    data = data[keep_index]
    
    # 股票代码顺序重置
    data.sort_index(inplace=True)
    
    # 调整上市日期的格式
    data["S_INFO_LISTDATE"] = pd.to_datetime(data["S_INFO_LISTDATE"])
    
    # 保存数据
    data.to_pickle("data/basic/stock_info")
    stock_info = pd.read_pickle("data/basic/stock_info")
    
# =============================================================================
# 股票预处理相关数据
# =============================================================================
    
    # 股票收盘价、交易状态
    print("股票收盘价、交易状态")
    data_names = ["S_DQ_ADJCLOSE","S_DQ_TRADESTATUS"]
    df_list = model.get_specific_data(database=model.stock_database,
                                      collection='AShareEODPrices',
                                      start_date=start_date,
                                      end_date=end_date,
                                      date_name='TRADE_DT',
                                      stock_name='S_INFO_WINDCODE',
                                      target_list=data_names)
    for i,name in enumerate(data_names):
        if df_list[i] is not None:
            df_list[i] = df_list[i].reindex(columns=stock_info.index)
            df_list[i].to_pickle("data/daily/stock/"+name)
        else:
            print("数据缺失:",name)
    
    # 股票涨跌停状态、换手率、总市值
    print("涨跌停状态、换手率、总市值")
    data_names = ["UP_DOWN_LIMIT_STATUS","S_DQ_TURN"]
    df_list = model.get_specific_data(database=model.stock_database,
                                      collection="AShareEODDerivativeIndicator",
                                      start_date=start_date,
                                      end_date=end_date,
                                      date_name='TRADE_DT',
                                      stock_name='S_INFO_WINDCODE',
                                      target_list=data_names)
    for i,name in enumerate(data_names):
        if df_list[i] is not None:
            df_list[i] = df_list[i].reindex(columns=stock_info.index)
            df_list[i].to_pickle("data/daily/stock/"+name)
        else:
            print("数据缺失:",name)
            
    data_names = ["S_DQ_MV"]
    df_list = model.get_specific_data(database=model.stock_database,
                                      collection="AShareEODDerivativeIndicator",
                                      start_date=start_date,
                                      end_date=end_date,
                                      date_name='TRADE_DT',
                                      stock_name='S_INFO_WINDCODE',
                                      target_list=data_names)
    for i,name in enumerate(data_names):
        if df_list[i] is not None:
            df_list[i] = 10000*df_list[i].reindex(columns=stock_info.index)#市值单位为（万元）,需要乘10000
            df_list[i].to_pickle("data/daily/stock/"+name)
        else:
            print("数据缺失:",name)
            
    # ST股票数据
    print("ST状态")
    data_names = ["ST_mark"]
    df_list = model.get_specific_data(database=model.stock_database,
                                      collection='AShareST',
                                      start_date=start_date,
                                      end_date=end_date,
                                      date_name='date',
                                      stock_name='S_INFO_WINDCODE',
                                      target_list=data_names)
    for i,name in enumerate(data_names):
        if df_list[i] is not None:
            df_list[i] = df_list[i].reindex(columns=stock_info.index)
            df_list[i].to_pickle("data/daily/stock/"+name)
        else:
            print("数据缺失:",name)
    
# =============================================================================
# 计算日频交易日序列
# =============================================================================
    
    # 读取日频序列
    print("交易日序列")
    daily_dates_index = pd.read_pickle('data/daily/stock/S_DQ_ADJCLOSE').index

    # 生成Series
    daily_dates = pd.Series(daily_dates_index,index=daily_dates_index)
     
    # 数据存储
    daily_dates.to_pickle('data/basic/daily_dates')
    daily_dates = pd.read_pickle('data/basic/daily_dates')
    
# =============================================================================
# 计算上市日期
# =============================================================================
    
    # 计算股票上市日期
    print("股票上市天数")

    stock_codes = stock_info.index
    
    # 展开上市日期序列
    list_date = pd.DataFrame(np.tile(pd.to_datetime(stock_info.loc[:,"S_INFO_LISTDATE"]),
                                      (daily_dates.shape[0],1)),
                              index=daily_dates,columns=stock_codes)
    
    # 展开交易日序列
    daily_date = pd.DataFrame(np.tile(daily_dates,(len(stock_codes),1)),
                              index=stock_codes,columns=daily_dates).T
    
    # 计算退市日期
    listed_days = daily_date-list_date
    
    # 数据存储
    listed_days.to_pickle('data/daily/stock/listed_days')
    
# =============================================================================
# 股票预处理
# =============================================================================
    
    # 读取股票预处理相关数据
    print("股票预处理")
    turn = pd.read_pickle("data/daily/stock/S_DQ_TURN")
    updown = pd.read_pickle("data/daily/stock/UP_DOWN_LIMIT_STATUS")
    trade_st = pd.read_pickle("data/daily/stock/S_DQ_TRADESTATUS")
    st_mark = pd.read_pickle("data/daily/stock/ST_mark")
    listed_days  = pd.read_pickle("data/daily/stock/listed_days")
    
    # 过滤器：滤除涨跌停、换手率为0、非交易状态、ST状态、上市天数小于180的股票
    stock_filter = (updown.abs() != 1) & (turn > 1e-8) & \
                    (trade_st == "交易") & (st_mark != 1) & \
                    (listed_days >= pd.Timedelta("180 days"))
    
    stock_filter.to_pickle("data/daily/stock/stock_filter")

# =============================================================================
# 指数收盘价
# =============================================================================

    print("基准指数收盘价")
    data_names = ["S_DQ_CLOSE"]
    df_list = model.get_specific_data(database=model.index_database,
                                      collection="AIndexEODPrices",
                                      start_date=start_date,
                                      end_date=end_date,
                                      date_name="TRADE_DT",
                                      stock_name="S_INFO_WINDCODE",
                                      target_list=data_names,
                                      stock_code=["000300.SH"])
    for i,name in enumerate(data_names):
        if df_list[i] is not None:
            df_list[i].to_pickle("data/daily/index/"+name)
        else:
            print("数据缺失:",name)
            
# =============================================================================
# 行业信息读取
# =============================================================================
    
    # 中信行业信息
    indus_info = pd.read_excel('data/basic/行业代码汇总.xlsx', index_col=0, sheet_name='映射关系')
    
    # 华泰行业划分
    target1_indus = indus_info[indus_info['类别'].str.contains('一级行业')]
    target1_indus.to_pickle('data/basic/indus1_info')    
    
    # target2_indus = indus_info[indus_info['类别'].str.contains('二级行业')]
    # target2_indus.to_pickle('data/basic/indus2_info')
    
    # target3_indus = indus_info[indus_info['类别'].str.contains('三级行业')]
    # target3_indus = target3_indus[target3_indus['Wind代码'].str.contains('旧版代码沿用') == False]
    # target3_indus.to_pickle('data/basic/indus3_info')
    
    # 底层库行业代码替换成行业名称
    name_code_to_name = dict(zip(indus_info['行业代码'], indus_info['行业名称']))
    name_code_to_name['nan'] = 'nan'
    
    # Wind底层库行业代码替换成行业名称
    name_windcode_to_name = dict(zip(indus_info['Wind代码'], indus_info['行业名称']))
    name_windcode_to_name['nan'] = 'nan'    


# =============================================================================
# 行业归属
# =============================================================================
    
    print("中信行业归属")
    
    # # 中信一级行业归属
    indus1_belong = model.get_specific_data(database = model.indus_database,
                                      collection = 'AShareIndustriesClassCITICS',
                                      start_date = start_date,
                                      end_date = end_date,
                                      date_name = 'date',
                                      stock_name = 'stock_code',
                                      target_list = ['cs_indus1_code']) 
    
    # # 中信二级行业归属
    # indus2_belong = model.get_specific_data(database = model.indus_database,
    #                                   collection = 'AShareIndustriesClassCITICS',
    #                                   start_date = start_date,
    #                                   end_date = end_date,
    #                                   date_name = 'date',
    #                                   stock_name = 'stock_code',
    #                                   target_list = ['cs_indus2_code'])

    # # 中信三级行业归属
    # indus3_belong = model.get_specific_data(database = model.indus_database,
    #                                   collection = 'AShareIndustriesClassCITICS',
    #                                   start_date = start_date,
    #                                   end_date = end_date,
    #                                   date_name = 'date',
    #                                   stock_name = 'stock_code',
    #                                   target_list = ['cs_indus3_code'])
    
    # 底层库代码替换成行业名称
    indus1_belong = indus1_belong[0].astype('str')    
    indus1_belong_name = indus1_belong.applymap(lambda x: name_code_to_name[x])
    
    # indus2_belong = indus2_belong[0].astype('str')    
    # indus2_belong_name = indus2_belong.applymap(lambda x: name_code_to_name[x])    

    # indus3_belong = indus3_belong[0].astype('str')    
    # indus3_belong_name = indus3_belong.applymap(lambda x: name_code_to_name[x]) 

    # 行业归属数据存储
    indus1_belong_name = indus1_belong_name.reindex(columns=stock_info.index)
    indus1_belong_name.to_pickle('data/daily/stock/indus1_belong')
    
    # # 行业归属数据存储
    # indus2_belong_name = indus2_belong_name.reindex(columns=stock_info.index) 
    # indus2_belong_name.to_pickle('data/daily/stock/indus2_belong')
    
    # # 行业归属数据存储
    # indus3_belong_name = indus3_belong_name.reindex(columns=stock_info.index) 
    # indus3_belong_name.to_pickle('data/daily/stock/indus3_belong')
    
# =============================================================================
# 中国A股资产负债表  AShareBalanceSheet
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
    
    print("资产负债表")
    data_names = ['TOT_ASSETS','MONETARY_CAP','ACCT_RCV','NOTES_RCV','TRADABLE_FIN_ASSETS',
                  'INVENTORIES','FIX_ASSETS','TOT_CUR_ASSETS','ST_BORROW','NOTES_PAYABLE','ACCT_PAYABLE',
                  'NON_CUR_LIAB_DUE_WITHIN_1Y','TOT_SHRHLDR_EQY_EXCL_MIN_INT','TOT_LIAB','TOT_CUR_LIAB']
    
    # 读取408001000合并报表数据，部分被处罚公司此部分数据会被后来的公告修正
    df_list = model.get_specific_data(database=model.stock_financial_database,
                                      collection="AShareBalanceSheet",
                                      start_date=start_date,
                                      end_date=end_date,
                                      date_name="REPORT_PERIOD",
                                      stock_name="S_INFO_WINDCODE",
                                      target_list=data_names,
                                      conditions={"STATEMENT_TYPE":408001000})
    
    # 读取408005000合并报表(更正前)数据，此部分数据是公司最原始的数据
    original_df_list = model.get_specific_data(database=model.stock_financial_database,
                                      collection="AShareBalanceSheet",
                                      start_date=start_date,
                                      end_date=end_date,
                                      date_name="REPORT_PERIOD",
                                      stock_name="S_INFO_WINDCODE",
                                      target_list=data_names,
                                      conditions={"STATEMENT_TYPE":408005000})
            
    for i,name in enumerate(data_names):
        if (original_df_list[i] is not None) & (df_list[i] is not None):
            # 数据替换，有旧数据就优先使用旧数据
            rep_locs = ~original_df_list[i].isnull()
            df_list[i][rep_locs] = original_df_list[i][rep_locs]
            
            df_list[i] = df_list[i].reindex(columns=stock_info.index)
            df_list[i].to_pickle("data/quarterly/stock/"+name)
        
        elif df_list[i] is not None:
            df_list[i] = df_list[i].reindex(columns=stock_info.index)
            df_list[i].to_pickle("data/quarterly/stock/"+name)
        elif original_df_list[i] is not None:
            original_df_list[i].to_pickle("data/quarterly/stock/"+name)
        else:
            print("数据缺失:",name)
       
# =============================================================================a
#  中国A股现金流量表  AShareCashFlow
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

    print("现金流量表")
    data_names = ['CASH_RECP_SG_AND_RS','CASH_PAY_ACQ_CONST_FIOLTA','CASH_RECP_DISP_WITHDRWL_INVEST',
                  'CASH_RECP_RETURN_INVEST','NET_CASH_FLOWS_INV_ACT','CASH_RECP_CAP_CONTRIB',
                  'NET_CASH_FLOWS_FNC_ACT','DEPR_FA_COGA_DPBA','AMORT_INTANG_ASSETS','AMORT_LT_DEFERRED_EXP',
                  'CASH_CASH_EQU_BEG_PERIOD','CASH_CASH_EQU_END_PERIOD','FREE_CASH_FLOW','NET_INCR_CASH_CASH_EQU',
                  'NET_CASH_FLOWS_OPER_ACT']
    
    # 读取408001000合并报表数据，部分被处罚公司此部分数据会被后来的公告修正
    df_list = model.get_specific_data(database=model.stock_financial_database,
                                      collection="AShareCashFlow",
                                      start_date=start_date,
                                      end_date=end_date,
                                      date_name="REPORT_PERIOD",
                                      stock_name="S_INFO_WINDCODE",
                                      target_list=data_names,
                                      conditions={"STATEMENT_TYPE":408001000})
    
    # 读取408005000合并报表(更正前)数据，此部分数据是公司最原始的数据
    original_df_list = model.get_specific_data(database=model.stock_financial_database,
                                      collection="AShareCashFlow",
                                      start_date=start_date,
                                      end_date=end_date,
                                      date_name="REPORT_PERIOD",
                                      stock_name="S_INFO_WINDCODE",
                                      target_list=data_names,
                                      conditions={"STATEMENT_TYPE":408005000})
            
    for i,name in enumerate(data_names):
        if (original_df_list[i] is not None) & (df_list[i] is not None):
            # 数据替换，有旧数据就优先使用旧数据
            rep_locs = ~original_df_list[i].isnull()
            df_list[i][rep_locs] = original_df_list[i][rep_locs]
            
            df_list[i] = df_list[i].reindex(columns=stock_info.index)
            df_list[i].to_pickle("data/quarterly/stock/"+name)
        
        elif df_list[i] is not None:
            df_list[i] = df_list[i].reindex(columns=stock_info.index)
            df_list[i].to_pickle("data/quarterly/stock/"+name)
        elif original_df_list[i] is not None:
            original_df_list[i].to_pickle("data/quarterly/stock/"+name)
        else:
            print("数据缺失:",name)
      
        
# =============================================================================
# 中国A股利润表  AShareIncome
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
    
    print("利润表")
    data_names = ['ANN_DT','OPER_REV', 'OPER_PROFIT', 'TOT_PROFIT', 'NET_PROFIT_EXCL_MIN_INT_INC',
                  'NET_PROFIT_INCL_MIN_INT_INC', 'TOT_OPER_COST', 'LESS_OPER_COST','TOT_OPER_REV',
                  'LESS_SELLING_DIST_EXP','LESS_GERL_ADMIN_EXP','LESS_FIN_EXP',
                  'STMNOTE_FINEXP','RD_EXPENSE','INC_TAX','COMSHARE_DVD_PAYABLE','EBIT','EBITDA']
    
    # 读取408001000合并报表数据，部分被处罚公司此部分数据会被后来的公告修正
    df_list = model.get_specific_data(database=model.stock_financial_database,
                                      collection="AShareIncome",
                                      start_date=start_date,
                                      end_date=end_date,
                                      date_name="REPORT_PERIOD",
                                      stock_name="S_INFO_WINDCODE",
                                      target_list=data_names,
                                      conditions={"STATEMENT_TYPE":408001000})
    
    # 读取408005000合并报表(更正前)数据，此部分数据是公司最原始的数据
    original_df_list = model.get_specific_data(database=model.stock_financial_database,
                                      collection="AShareIncome",
                                      start_date=start_date,
                                      end_date=end_date,
                                      date_name="REPORT_PERIOD",
                                      stock_name="S_INFO_WINDCODE",
                                      target_list=data_names,
                                      conditions={"STATEMENT_TYPE":408005000})
    
    for i,name in enumerate(data_names):
        if (original_df_list[i] is not None) & (df_list[i] is not None):
            # 数据替换，有旧数据就优先使用旧数据
            rep_locs = ~original_df_list[i].isnull()
            df_list[i][rep_locs] = original_df_list[i][rep_locs]
            
            df_list[i] = df_list[i].reindex(columns=stock_info.index)
            df_list[i].to_pickle("data/quarterly/stock/"+name)
        
        elif df_list[i] is not None:
            df_list[i] = df_list[i].reindex(columns=stock_info.index)
            df_list[i].to_pickle("data/quarterly/stock/"+name)
        elif original_df_list[i] is not None:
            original_df_list[i].to_pickle("data/quarterly/stock/"+name)
        else:
            print("数据缺失:",name)
    
    
# =============================================================================
# 中国A股财务指标  AShareFinancialIndicator
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
    
    print("财务指标")
    data_names = ['S_FA_YOYEPS_BASIC','S_FA_YOYOP','S_FA_YOYEBT',
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
                  'S_FA_OCFTOPROFIT','S_FA_CASHTOLIQDEBTWITHINTEREST']
    
    df_list = model.get_specific_data(database=model.stock_financial_database,
                                      collection="AShareFinancialIndicator",
                                      start_date=start_date,
                                      end_date=end_date,
                                      date_name="REPORT_PERIOD",
                                      stock_name="S_INFO_WINDCODE",
                                      target_list=data_names)
    
    for i,name in enumerate(data_names):
        if df_list[i] is not None:
            df_list[i] = df_list[i].reindex(columns=stock_info.index)
            df_list[i].to_pickle("data/quarterly/stock/"+name)
        else:
            print("数据缺失:",name)
    
    # 单独计算S_FA_ROE_YOY
    data = df_list[data_names.index("S_FA_ROE")]
    data_yoy = data - data.shift(4) 
    data_yoy = data_yoy.reindex(columns=stock_info.index)
    data_yoy.to_pickle('data/quarterly/stock/S_FA_ROE_YOY')

# =============================================================================
# 个股层面一致预期数据
# =============================================================================
    
    # 数据类型
    types = ['FY2','CAGR']
    
    data_names = ["EST_EPS","EST_ROE","NET_PROFIT","EST_OPER_REVENUE","EST_CFPS",
                  "EST_DPS","EST_BPS","EST_EBIT","EST_EBITDA","EST_TOTAL_PROFIT",
                  "EST_OPER_PROFIT"]
    
    for data_type in types:
        print("个股一致预期"+data_type)
        
        df_list = model.get_specific_data(database=model.est_database,
                                          collection="AShareConsensusRollingData",
                                          start_date=start_date,
                                          end_date=end_date,
                                          date_name="EST_DT",
                                          stock_name="S_INFO_WINDCODE",
                                          target_list=data_names,
                                          conditions={"ROLLING_TYPE":data_type})
        
        for i,name in enumerate(data_names):
            if df_list[i] is not None:
                # 用前期数据进行填充
                df_list[i].ffill(inplace=True)
                df_list[i] = df_list[i].reindex(columns=stock_info.index)
                df_list[i].to_pickle("data/daily/stock/"+name+"_"+data_type)
            else:
                print("数据缺失:",name)
# =============================================================================
# 指数收盘价
# =============================================================================

    print("基准指数收盘价")
    data_names = ["S_DQ_CLOSE"]
    index_list = ["000001.SH","000002.SH","000016.SH","000300.SH","399001.SZ",
                  "399006.SZ","000905.CSI","000985.CSI"]
    df_list = model.get_specific_data(database=model.index_database,
                                      collection="AIndexEODPrices",
                                      start_date=start_date,
                                      end_date=end_date,
                                      date_name="TRADE_DT",
                                      stock_name="S_INFO_WINDCODE",
                                      target_list=data_names,
                                      stock_code=index_list)
    for i,name in enumerate(data_names):
        if df_list[i] is not None:
            df_list[i].to_pickle("data/daily/index/"+name)
        else:
            print("数据缺失:",name)


# =============================================================================
# 股票预处理相关数据
# =============================================================================
    
    print("流通市值")
    data_names = ["S_VAL_MV"]
    df_list = model.get_specific_data(database=model.stock_database,
                                      collection="AShareEODDerivativeIndicator",
                                      start_date=start_date,
                                      end_date=end_date,
                                      date_name='TRADE_DT',
                                      stock_name='S_INFO_WINDCODE',
                                      target_list=data_names)
    for i,name in enumerate(data_names):
        if df_list[i] is not None:
            df_list[i] = 10000*df_list[i].reindex(columns=stock_info.index)
            df_list[i].to_pickle("data/daily/stock/"+name)
        else:
            print("数据缺失:",name)
