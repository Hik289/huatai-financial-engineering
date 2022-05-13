# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 21:14:58 2018


"""

from utils import DataUtils
import pandas as pd
import numpy as np
import scipy.io as sio
import pymongo

# -----------------------------------------------------------------------------
# 基本面因子构建的基类，主要提供一些通用方法
# -----------------------------------------------------------------------------
class BaseClass(object):
    
    # -------------------------------------------------------------------------
    # 实例化对象，读取按个股合成行业时所以来的必须数据，避免重复加载
    # -------------------------------------------------------------------------
    def __init__(self):
        
        # 所有股票信息
        self.stock_info = DataUtils.get_basic_info('stock_info')
        self.stock_id = list(self.stock_info.index)
        self.stock_name = list(self.stock_info['S_INFO_NAME'])
        
        # 所有行业信息
        self.indus_info = DataUtils.get_basic_info('indus_info')
        self.indus_code = list(self.indus_info.index)
        self.indus_name = list(self.indus_info['行业名称'])   
        
        # 季报公布日期(季频)
        self.issue_date = DataUtils.get_quarterly_info('stock', 'ANN_DT')
        self.issue_date = self.issue_date.reindex(columns=self.stock_id)
        self.issue_date = self.issue_date.apply(lambda x : pd.to_datetime(x))
        
        # 行业归属数据，需要按个股上市、退市日期重置无效数据
        self.indus_belong =  DataUtils.get_daily_info('stock', 'indus_belong')
        self.indus_belong = self.indus_belong.reindex(columns=self.stock_id)

        # 获取一级行业收盘价
        self.indus_close = DataUtils.get_daily_info('indus', 'close')
        
        # 所有日频交易日期序列
        self.daily_dates = pd.Series(self.indus_close.index, index=self.indus_close.index)
        
        
    # -------------------------------------------------------------------------
    # 计算季报日期对应的最晚法定公布日期，输入需要是Timestamp格式
    # -------------------------------------------------------------------------
    def get_deadline_date(self, q_date):
        
        # 如果是年报，则年索引加1
        year = q_date.year + 1 if q_date.month == 12 else q_date.year
        
        # 获取季报对应的月份日期
        month_to_date = {3 : '0430', 6 : '0831', 9 : '1031', 12 : '0430'}
        
        # 返回拼装后的日期
        return pd.to_datetime(str(year) + month_to_date[q_date.month])
   
    
    # -------------------------------------------------------------------------
    # 计算当前能看到的最新季报日期
    # 4月30日之前，至少应该能看到去年的三季报
    # 5月1日至8月31日，至少应该看到今年一季报和年报
    # 9月1日至10月31日，至少应该看到今年的半年报
    # 11月1日至年底，至少应该看到今年三季报
    # -------------------------------------------------------------------------
    def get_latest_visible_quarter_date(self, panel_date):
                
        # 根据当前年份计算几个关键时点
        th1 = pd.to_datetime(str(panel_date.year) + '0430')
        th2 = pd.to_datetime(str(panel_date.year) + '0831')
        th3 = pd.to_datetime(str(panel_date.year) + '1031')
        
        # 根据截面日期进行相应判断
        if panel_date <= th1:
            quarter_date = pd.to_datetime(str(panel_date.year-1) + '0930')
        elif panel_date > th1 and panel_date <= th2:
            quarter_date = pd.to_datetime(str(panel_date.year) + '0331')
        elif panel_date > th2 and panel_date <= th3:
            quarter_date = pd.to_datetime(str(panel_date.year) + '0630')
        else:
            quarter_date = pd.to_datetime(str(panel_date.year) + '0930')
    
        return quarter_date.strftime('%Y-%m-%d')
    
    
    # -------------------------------------------------------------------------
    # 计算资产负债项的期初、期末平均值
    # 如果是TTM场景，期初为去年同期
    # 如果是QFA场景，期初为上个季度
    # 如果是原始累计值，期初为去年年报
    # -------------------------------------------------------------------------
    def transform_avg(self, df_factor, transform_method):
        
        # 生成季频日期
        quarterly_dates = list(df_factor.index)
        
        # 目标截面日期区间
        panel_dates = df_factor.loc['2001':, :].index
        
        # 定义期初、期末净资产求平均的函数，这里考虑期初、期末有一期为空时用另一值替代
        avg_func = lambda x,y: (x.fillna(y) + y.fillna(x))/2
        
        # 初始化返回值
        df_ret = pd.DataFrame(index=panel_dates, columns=df_factor.columns, dtype=float)
        
        # 遍历每一个截面
        for date in panel_dates:
        
            # 期初报告移位值，如果ttm则取一年前，如果是sqf则取上一季，否则为上年年报
            if transform_method == 'ttm':
                q_shift = 4
            elif transform_method == 'qfa':
                q_shift = 1
            else:
                q_shift = (int(date.month / 3) - 1) % 4 + 1
         
            # 获取期初如期
            last_date = quarterly_dates[(quarterly_dates.index(date))-q_shift]
            
            # 期初、期末求平均
            avg_ret = avg_func(df_factor.loc[date, :],df_factor.loc[last_date, :])
            
            # 存储结果
            df_ret.loc[date, :] = avg_ret.astype(float)
        
        return df_ret


    # -------------------------------------------------------------------------
    # 按照wind的方式来执行利润表或现金流量表科目的转换：
    # 1、年报TTM=年报，季报TTM=当季值+上年年报值-去年同季值
    # 2、如果当季值或去年同期值有空值，则直接取去年年报值作为当前TTM值
    # -------------------------------------------------------------------------
    def transform_ttm_in_wind_way(self, df_factor):
        
#        df_factor = factor_input.copy()
#        df_factor[df_factor.isnull()] = 0
        
        # 季频日期序列
        quaterly_dates = list(df_factor.index)
        
        # 目标截面日期区间（为了给同比计算保留裕量，这里从04年开始）
        panel_dates = df_factor.loc['2001':, :].index
        
        # 初始化返回值
        df_ret = pd.DataFrame(index=panel_dates, columns=df_factor.columns, dtype=float)
        
        # 遍历每一个截面
        for date in panel_dates:
            
            # 当前日期在季报序列中的位置
            q_loc = quaterly_dates.index(date)
            
            # 获取当前季报编号，并执行不同操作
            q_index = int(date.month/3) % 4            
            if q_index == 0:
                # 如果是年报值，直接设置
                df_ret.loc[date, :] = df_factor.loc[date,:]
            else:
                # 如果是季报值，当前数据 + 去年年报 - 去年同期
                data_ttm = df_factor.iloc[q_loc-q_index,:] + \
                        df_factor.iloc[q_loc,:] - df_factor.iloc[q_loc-4,:]
                        
                # 计算出有空值则直接用去年年报再填充一遍
                data_ttm = data_ttm.fillna(df_factor.iloc[q_loc-q_index,:])
                
                # 保存结果
                df_ret.loc[date, :] = data_ttm
        
        return df_ret
     
        
    # -------------------------------------------------------------------------
    # 按照wind的方式来执行利润表或现金流量表科目的转换：
    # 1、一季报qfa=当季值，其他季报qfa=当季值-上季值
    # 2、如果当季值或去年同期值有空值，则直接取去年年报值作为当前TTM值
    # -------------------------------------------------------------------------
    def transform_qfa_in_wind_way(self, df_factor):
        
        # 季频日期序列
        quaterly_dates = list(df_factor.index)
        
        # 目标截面日期区间
        panel_dates = df_factor.loc['2005':, :].index
        
        # 初始化返回值
        df_ret = pd.DataFrame(index=panel_dates, columns=df_factor.columns, dtype=float)
        
        # 遍历每一个截面
        for date in panel_dates:
            
            # 当前日期在季报序列中的位置
            q_loc = quaterly_dates.index(date)
            
            # 获取当前季报编号，并执行不同操作
            q_index = int(date.month/3) % 4            
            if q_index == 1:
                # 如果是一季报，直接设置
                df_ret.loc[date, :] = df_factor.loc[date,:]
            else:
                # 如果是其他季报，则取当季值减去上季值，且上季值的空值用0替代
                df_ret.loc[date, :] = df_factor.iloc[q_loc,:] - df_factor.iloc[q_loc-1,:].fillna(0)
        
        return df_ret
        
    
    # -------------------------------------------------------------------------
    # 根据个股合成的行业汇总数据，重构成可直接用于回测的因子暴露，主要执行：
    # 1、将原始数据中的季末日期调整为法定最晚财报公布日对应的交易日期
    # 2、去除年报数据，因为年报和一季报的最晚公布日期相同，采用最新的结果
    # [输入]
    # df_factor：    原始合成数据，标准时间序列，行索引为截面日期，列索引为行业名称
    #                注意索引列的日期对应标准的季末日期
    # [返回]
    # factor_expo：  重构后的因子暴露，索引列变成法定最晚公布日对应的交易日期
    # -------------------------------------------------------------------------
    def format_quarterly_expo(self, df_factor):
        
        # 初始化返回值，注意这里需要以拷贝的方式，否则更改会直接作用在输入值上
        factor_expo = df_factor.copy()
        
        # 将日频交易日期重采样至月频
        monthly_dates = self.daily_dates.resample('M').last()
        
        # 将季末日期映射为法定最晚公布日期
        factor_expo.index = [self.get_deadline_date(date) for date in factor_expo.index]
        
        # 将法定最晚公布日期映射成最近的交易日
        factor_expo.index = monthly_dates.reindex(index=factor_expo.index)
        
        # 去重，相当于不使用年报数据
        factor_expo = factor_expo.groupby(factor_expo.index).last()
                
        return factor_expo
        
    
    # -------------------------------------------------------------------------
    # 将因子构建结果生成符合MATLAB回测平台要求的数据格式，并写入MAT文件
    # [输入]
    # factor_expo：  因子暴露，标准时间序列，行索引为截面日期，列索引为行业编号
    # factor_sign：  因子方向，只能是1或者-1
    # factor_name：  因子名称，也是文件存储名称
    # -------------------------------------------------------------------------
    def write_to_mat(self, factor_expo, factor_sign, factor_name):
                 
        # 格式化因子方向
        factor_sign = np.array([factor_sign])
        
        # 格式化行业列表
        target_indus = np.array([factor_expo.columns],dtype="O").T
        
        # 格式化截面日期，这里需要转换为MATLAB的日期格式
        panel_dates = factor_expo.index;
            
        #  统一转换日期为适用于Matlab端的距离特定日期的天数
        start_date_num = 700000;
        start_date = pd.Timestamp('1916-07-14')
        panel_dates = panel_dates-start_date
        panel_dates = [date.days + start_date_num for date in panel_dates]
        panel_dates = np.array([panel_dates], dtype='float64').T
        
        # 格式化因子暴露矩阵  
        factor_expos = factor_expo.values
        
        # 生成最终结果
        result = {'factor_sign' : factor_sign,
                  'target_indus' : target_indus,
                  'panel_dates' : panel_dates,
                  'factor_expo' : factor_expos}
        
        # 写入MAT文件          
        sio.savemat('result/{0}.mat'.format(factor_name), {factor_name : result})   
         
        
        
        