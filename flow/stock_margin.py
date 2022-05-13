
import sys

import os 
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# 两融资金指标计算
# -----------------------------------------------------------------------------
class margin_money():

    # -------------------------------------------------------------------------
    # 实例化对象，主要用于加载全局数据，避免每次重复加载
    # -------------------------------------------------------------------------
    def __init__(self):
        
        # 获取上级文件路径
        file_path = os.path.abspath(os.path.dirname(os.getcwd()))     
            
        # 读取A股均价
        self.avg_price = pd.read_pickle(file_path + '/flow/data/daily/stock/S_DQ_AVGPRICE')
                
        # 读取A股成交额
        self.stock_amount = pd.read_pickle(file_path + '/flow/data/daily/stock/S_DQ_AMOUNT').loc['2005-01-01':,:] * 1000
        
        # 读取A股流通市值
        self.float_size = pd.read_pickle(file_path + '/flow/data/daily/stock/S_DQ_MV').loc['2005-01-01':,:] * 10000
            
        # 字典映射关系
        self.cn_dict = {
            "融资融券余额" : 'balance', # S_MARGIN_MARGINTRADEBALANCE
            "融资余额" : 'tr_balance', # S_MARGIN_TRADINGBALANCE
            "融券余额" : 'slend_balance', # S_MARGIN_SECLENDINGBALANCE
            "融资买入额" : 'borrow',  #  S_MARGIN_PURCHWITHBORROWMONEY
            "融资偿还额" : 'repay',  # S_MARGIN_REPAYMENTTOBROKER
            "融券偿还量" : 'borrow_sec',  # S_MARGIN_REPAYMENTOFBORROWSEC
            "融券卖出量" : 'repay_sec', # S_MARGIN_SALESOFBORROWEDSEC
            "净融资" : 'net_buy',
            "净融券" : 'net_sell'}
 
        # 两融数据读取
        self.borrow = pd.read_pickle(file_path + '/flow/data/daily/stock/S_MARGIN_PURCHWITHBORROWMONEY')
        self.repay = pd.read_pickle(file_path + '/flow/data/daily/stock/S_MARGIN_REPAYMENTTOBROKER')
        self.borrow_sec = pd.read_pickle(file_path + '/flow/data/daily/stock/S_MARGIN_REPAYMENTOFBORROWSEC')
        self.repay_sec = pd.read_pickle(file_path + '/flow/data/daily/stock/S_MARGIN_SALESOFBORROWEDSEC')
        self.balance = pd.read_pickle(file_path + '/flow/data/daily/stock/S_MARGIN_MARGINTRADEBALANCE')
        self.tr_balance = pd.read_pickle(file_path + '/flow/data/daily/stock/S_MARGIN_TRADINGBALANCE')  # 融资余额
        self.slend_balance = pd.read_pickle(file_path + '/flow/data/daily/stock/S_MARGIN_SECLENDINGBALANCE')  # 融券余额

        # 其他数据计算
        self.net_buy = self.borrow - self.repay
        self.net_sell = self.borrow_sec - self.repay_sec

        # 所有日频交易日期序列
        self.daily_dates = pd.Series(self.avg_price.index, index=self.avg_price.index)
        
        
    # -------------------------------------------------------------------------
    # 依据设定时间间隔获取设定区间内的交易日序列
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
    # 数据按设定频率进行提取
    # freq: 'W' 或是 'M'
    # -------------------------------------------------------------------------
    def data_resample(self, origin_data, freq, method):
                
        if method == 'sum':
                
            # 取不同频率数据
            data_freq = origin_data.resample(freq).sum()
            
        elif method == 'last':
                                
            # 取不同频率数据
            origin_data_change = origin_data.loc[origin_data.sum(axis=1)!=0, :]        
            data_freq = origin_data_change.resample(freq).last()
                
        # 日期调整，汇总到最近的交易日上
        change_dates = self.daily_dates.resample(freq).last()
        
        # 替换名称
        data_freq.index = pd.to_datetime(change_dates[data_freq.index].values)
        
        # 剔除空值数据
        data_freq = data_freq.loc[data_freq.index.notnull(), :]
        
        return data_freq
    
    
    # -------------------------------------------------------------------------
    # 存量视角：分析资金总量变化情况
    # -------------------------------------------------------------------------
    def gen_money_holdings(self, data_name, data_type='normal', freq='W', method='qoq'):

        # 原始资金数据读取
        if data_name in ["融券余额", "融资偿还额"]: # 指标逻辑为负
            origin_data = - eval('self.{}'.format(self.cn_dict[data_name]))
        else:
            origin_data = eval('self.{}'.format(self.cn_dict[data_name]))
             
        # 归一化
        if data_type == 'float':
            
            # 读取流通市值，计算资金流数据和行业流通市值之比
            indus_float = self.float_size
            indus_float[indus_float==0] = np.nan
            indus_data = self.data_resample(origin_data/indus_float, freq, 'last')
            
        if data_type == 'amount':
            
            # 读取成交额，计算资金流数据和行业成交额之比
            indus_amount = self.stock_amount
            indus_amount[indus_amount==0] = np.nan
            indus_data = self.data_resample(origin_data/indus_amount, freq, 'last')
            
        elif data_type == 'orig':
                             
            # 取样后数据
            data_resample = self.data_resample(origin_data, freq, 'last')
            
            # 数据在全市场占比
            indus_data = data_resample.multiply(1 / data_resample.sum(axis=1), axis=0)
            
        # 根据同比/环比设定进行数据调整
        if method == 'orig':
            output = indus_data
        
        elif method == 'qoq':
            shift_sign = 1    
            output = indus_data - indus_data.shift(shift_sign)
            
        elif method == 'yoy':
            if freq == 'W':
                shift_sign = 52
            else:
                shift_sign = 12
            output = indus_data - indus_data.shift(shift_sign)
            
        return output

    
    # -------------------------------------------------------------------------
    # 增量视角：统计资金流入边际变化情况
    # -------------------------------------------------------------------------
    def gen_money_change(self, data_name, data_type='normal', freq='W', method='qoq'):
        
        # 原始资金数据读取
        if data_name in ["融券余额", "融资偿还额"]: # 指标逻辑为负
            origin_data = - eval('self.{}'.format(self.cn_dict[data_name]))
        else:
            origin_data = eval('self.{}'.format(self.cn_dict[data_name]))
                
            
        # 行业资金进行归一化
        data_resample = self.data_resample(origin_data, freq, 'sum')
        
        # 归一化
        if data_type == 'float':
                
            # 取流通市值数据，并按照不同频率汇总
            indus_float = self.float_size   
            indus_float[indus_float==0] = np.nan
            float_resample = self.data_resample(indus_float, freq, 'sum')
            
            # 将资金数据归一化
            indus_data = data_resample / float_resample
                        
        # 归一化
        if data_type == 'amount':
                
            # 取成交额数据，并按照不同频率汇总
            indus_amount = self.stock_amount       
            indus_amount[indus_amount==0] = np.nan
            float_resample = self.data_resample(indus_amount, freq, 'sum')
            
            # 将资金数据归一化
            indus_data = data_resample / float_resample
            
            
        elif data_type == 'orig':
            indus_data = data_resample
            pass

        
        # 根据同比/环比设定进行数据调整
        if method == 'orig':
            output = indus_data
        
        elif method == 'qoq':
            shift_sign = 1    
            output = indus_data - indus_data.shift(shift_sign)
            
        elif method == 'yoy':
            if freq == 'W':
                shift_sign = 52
            else:
                shift_sign = 12
            output = indus_data - indus_data.shift(shift_sign)
            
        return output

    
if __name__ == "__main__":
    
    model = margin_money()
        
    # 生成调仓信号
    for data_type in ['orig', 'float', 'amount']:     # 是否除以流通市值 
        for freq in ['W','M']:    # 信号发出频率
            for method in ['orig', 'qoq', 'yoy']:  # 信号调整
                
                print(data_type, freq, method)
            
                for data_name in ["融券余额", "融资余额"]:
                    
                    # 资金累计持有
                    data_holdings = model.gen_money_holdings(data_name, data_type, freq, method)                
                    data_holdings.to_pickle(file_path+'/flow/results/margin_{}_{}_{}_{}'.format(model.cn_dict[data_name], data_type, freq, method))
                

                for data_name in ["净融资","融资买入额","融资偿还额"]:
                    
                    # 资金边际变化
                    data_moneyflow = model.gen_money_change(data_name, data_type, freq, method)
                    data_moneyflow.to_pickle(file_path+'/flow/results/margin_{}_{}_{}_{}'.format(model.cn_dict[data_name], data_type, freq, method))
                
                
        
    # 融资余额指投资者每日融资买进与归还借款间的差额。融券余量金额指融券卖出额与每日偿还融券额的差额。
    
    # 本日融资融券余额＝本日融资余额＋本日融券余量金额
    # 本日融资余额＝前日融资余额＋本日融资买入额－本日融资偿还额；
    # 本日融资偿还额＝本日直接还款额＋本日卖券还款额＋本日融资强制平仓额＋本日融资正权益调整－本日融资负权益调整；
    # 本日融券余量=前日融券余量+本日融券卖出数量-本日融券偿还量；
    # 本日融券偿还量＝本日买券还券量＋本日直接还券量＋本日融券强制平仓量＋本日融券正权益调整－本日融券负权益调整－本日余券应划转量；
    # 融券单位：股（标的证券为股票）/份（标的证券为基金）/手（标的证券为债券）。
    # 明细信息中仅包含当前融资融券标的证券的相关数据，汇总信息中包含被调出标的证券范围的证券的余额余量相关数据。
    
    
    # 字典映射关系

        