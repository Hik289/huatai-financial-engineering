# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:56:20 2019


"""

from utils import BacktestUtils, PerfUtils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号


class model_backtest():
    
    # -------------------------------------------------------------------------
    # 实例化，加载基本信息
    # -------------------------------------------------------------------------
    def __init__(self):
            
        # 获取一级行业收盘价
        self.indus_close = pd.read_pickle('data/close')

        self.indus_info = pd.read_pickle('data/indus_info')
        
        self.daily_dates = pd.Series(self.indus_close.index, index=self.indus_close.index)
        
    # -------------------------------------------------------------------------
    # 获取设定时间区间内的月末交易日 - 设定时间间隔
    # [输入]
    # start_date     开始时间
    # end_date       终止时间
    # frequency      升采样频率，默认为一个月'M'，n个月为'nM'
    # -------------------------------------------------------------------------
    def gen_panel_dates(self, start_date, end_date, frequency='M'):
        
        # 指标时间区间
        month_end_dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
        
        # 将交易日序列按月频重采样，取最后一个值
        monthly_dates = self.daily_dates.resample('M').last()
        
        # 将月末自然日替换成月末交易日
        panel_dates = pd.to_datetime(monthly_dates[month_end_dates].values)
        
        return panel_dates
   
    
    # -------------------------------------------------------------------------
    # 回测过程计算
    # [输入]
    # df_factor      行业景气度指标，一般只有1，-1，0三个数值，分别代表多头/空头/空仓
    # start_date     计算回测净值曲线开始时间
    # end_date       计算回测净值曲线终止时间
    # -------------------------------------------------------------------------
    def backtest(self, df_factor_input, start_date, end_date, fee=0):
        
        df_factor = df_factor_input.copy()
        
        # 当行业景气度指标给定的最后一个日期大于最后一个交易日时（月末容易出现）
        # 最后一个交易信号无法调仓
        if df_factor.index[-1] >= self.daily_dates[-1]:
            df_factor = df_factor.drop(index=df_factor.index[-1])
            
        # 调仓日期为生成信号的下一天，即月初第一个交易日
        df_factor.index = [self.daily_dates[self.daily_dates > i].index[0] for i in df_factor.index]        
        
        # 根据输入的行业指标，计算多头和空头持仓
        long_portion_all = df_factor.copy() * 0
        short_portion_all = df_factor.copy() * 0
        
        # 遍历所有日期
        for date in df_factor.index:    
            
            # 一般情况，标为1的行业为多头持仓，这里简化为指标值大于零的行业为多头持仓
            if sum(df_factor.loc[date,:] > 0) != 0:
                long_portion_all.loc[date,df_factor.loc[date,:]>0] = 1/sum(df_factor.loc[date,:] > 0)
                
            # 一般情况，标为-1的行业为空头持仓，这里简化为指标值小于零的行业为空头持仓
            if sum(df_factor.loc[date,:] < 0) != 0:
                short_portion_all.loc[date,df_factor.loc[date,:]<0] = 1/sum(df_factor.loc[date,:] < 0)
                
        # 对持仓进行时间截断
        long_portion = long_portion_all[start_date:end_date]
        short_portion = short_portion_all[start_date:end_date]
        
        # 参照基准 - 行业收益率等权
        indus_return = self.indus_close.pct_change()
        base_return = indus_return.mean(axis=1) + 1
        base_close = base_return.cumprod()
        base_nav = base_close.loc[start_date:end_date]
            
            
        # 计算绝对净值        
        nav = pd.DataFrame(columns=['多头策略','空头策略','基准'])
        
        # 回测,计算策略净值
        nav['多头策略'], df_indus_return = BacktestUtils.cal_nav(long_portion, 
           self.indus_close[start_date:end_date], base_nav, fee)
            
        nav['空头策略'], df_indus_return_short = BacktestUtils.cal_nav(short_portion,
           self.indus_close[start_date:end_date], base_nav, fee)
        
        # 基准净值归一化
        nav['基准'] = base_nav / base_nav.values[0]
          
        # 计算相对净值
        nav_relative = pd.DataFrame(columns=['多头/基准','空头/基准'])
        nav_relative['多头/基准'] = nav['多头策略'] / nav['基准'] 
        nav_relative['空头/基准'] = nav['空头策略'] / nav['基准']
        
        # 返回绝对净值曲线，相对净值曲线，多头持仓
        return nav, nav_relative, long_portion


    # -------------------------------------------------------------------------
    # 根据复合行业景气度指标计算做多或是做空行业
    # [输入]
    # merge_factor    复合行业景气度指标
    # indus_num       通过复合行业指标计算持仓时，需要规定选择的行业个数，默认5个
    # drop_list       需要剔除的行业数目
    # -------------------------------------------------------------------------
    def cal_merge_ls_factor(self, merge_factor_input, indus_num=5, drop_list=[]):
                        
        # 舍弃部分行业
        merge_factor = merge_factor_input.copy()
        merge_factor.drop(columns=drop_list,inplace=True)
        self.indus_close.drop(columns=drop_list,inplace=True)
        
        # 对复合行业景气度指标进行指数加权
        merge_factor_ewm = merge_factor.ewm(alpha=0.99).mean()
        merge_factor_ewm.loc[:, '综合金融'] = np.nan
        merge_factor_ewm.loc[:'2011-01-01', '多元金融'] = np.nan

        # 加权后的景气度指标进行排序
        merge_factor_rank = merge_factor_ewm.rank(method='average', ascending=False, axis=1)

        # 初始化返回值
        merge_ls_factor = merge_factor.copy() * 0
               
        # 多头持仓行业
        long_judge = (merge_factor_rank <= indus_num)
        long_judge[long_judge.sum(axis=1) > indus_num] = (merge_factor_rank <= indus_num - 1)[long_judge.sum(axis=1) > indus_num] 
        merge_ls_factor[long_judge] = 1

        # 加权后的景气度指标进行排序 - 由小到大
        merge_factor_rank = merge_factor_ewm.rank(method='average', ascending=True, axis=1)

        # 空头持仓行业
        short_judge = (merge_factor_rank <= indus_num)
        short_judge[short_judge.sum(axis=1) > indus_num] = (merge_factor_rank <= indus_num - 1)[short_judge.sum(axis=1) > indus_num] 
        merge_ls_factor[short_judge] = - 1
        
        merge_ls_factor[merge_ls_factor.isnull()] = 0
                
        return merge_ls_factor
    
    
    
if __name__ == '__main__':        


# =============================================================================
#   单个指标测试
# =============================================================================
    
    # 模型初始化
    model = model_backtest()
    
    # 读取指标
    df_factor = pd.read_excel('data/滚动因子观点.xlsx', index_col=0)
    map_relation = pd.read_excel('data/映射关系.xlsx', index_col=0, sheet_name='映射关系汇总')
    
    # 行业数据
    indus_score = pd.DataFrame(0, index=df_factor.index, columns=map_relation.index)
    
    for index in df_factor.index:
        
        score = []
        
        if df_factor.loc[index,'增长'] == 1:
            score.append(map_relation.loc[:, '增长上行'])
        if df_factor.loc[index,'增长'] == -1:
            score.append(map_relation.loc[:, '增长下行'])
            
        if df_factor.loc[index,'通胀'] == 1:
            score.append(map_relation.loc[:, '通胀上行'])
        if df_factor.loc[index,'通胀'] == -1:
            score.append(map_relation.loc[:, '通胀下行'])
            
        if df_factor.loc[index,'信用'] == 1:
            score.append(map_relation.loc[:, '信用上行'])
        if df_factor.loc[index,'信用'] == -1:
            score.append(map_relation.loc[:, '信用下行'])
            
        if df_factor.loc[index,'货币'] == 1:
            score.append(map_relation.loc[:, '货币上行'])
        if df_factor.loc[index,'货币'] == -1:
            score.append(map_relation.loc[:, '货币下行'])
            
        if len(score) > 0:
            indus_score.loc[index,:] = pd.concat(score, axis=1).sum(axis=1)
    
    # 计算多空信号
    merge_ls_factor = model.cal_merge_ls_factor(indus_score, indus_num=5)
    
    # 回测
    nav, nav_relative, long_portion = model.backtest(merge_ls_factor, '2010-01-01', '2021-09-30')
        
    # 计算多头相对空头胜率
    perf = PerfUtils.excess_statis(nav['多头策略'], nav['空头策略'], long_portion.index)

    # 作图
    nav.plot()
    nav_relative.plot()






        
    