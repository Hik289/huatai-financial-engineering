# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 23:54:28 2021
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool

# =============================================================================
# 拥挤度指标计算主程序
# =============================================================================
class crowd_factors():
    
    def __init__(self):

        # 个股收盘价数据
        self.stock_close = pd.read_pickle("data/stock_close")
        self.stock_daily_ret = self.stock_close/self.stock_close.shift(1)-1
        
        # 指数收盘价数据
        self.indus_close = pd.read_pickle("data/Wind_indus_close")                
        self.index_daily_ret = self.indus_close.pct_change()
        
        # 计算行业指数超额收益净值
        index_excess = (self.index_daily_ret.T - self.index_daily_ret.mean(axis=1)).T + 1                
        self.index_exclose = index_excess.cumprod(skipna = True)
        self.index_exclose[np.isnan(self.index_daily_ret)] = np.nan
                
        # 行业均线
        self.MA_5 = self.indus_close.rolling(window = 5, min_periods = 2).mean()
        self.MA_10 = self.indus_close.rolling(window = 10,min_periods = 3).mean()
        self.MA_20 = self.indus_close.rolling(window = 20,min_periods = 5).mean()
        self.MA_60 = self.indus_close.rolling(window = 60,min_periods = 10).mean()
        
        # 股票换手率数据
        self.stock_turn = pd.read_pickle("data/stock_turn")
        self.stock_amount = pd.read_pickle("data/stock_amount")
        
        # 行业指数数据：成交量、成交额、换手率
        self.index_volume = pd.read_pickle("data/index_volume")
        self.index_amount = pd.read_pickle("data/index_amount")
        self.index_turn = pd.read_pickle("data/index_turn")

   
    # =========================================================================
    # 指标生成：动量指标
    #
    # [输入]
    # factor_name：指标名称
    # window：区间长度
    # =========================================================================
    def gen_indus_momentum(self, factor_name, window):
        

        # 普通动量
        if factor_name == "normal_momentum":
            
            # window日收益率
            factor = self.index_exclose.pct_change(window)
        
        # 路径调整动量
        if factor_name == "distance_momentum":
            
            # 每日收益率的绝对值
            daily_retabs = self.index_daily_ret.abs()
            
            # 区间求和计算绝对长度
            rolling_sum = daily_retabs.rolling(window).sum()
            
            factor = self.indus_close.pct_change(window) / rolling_sum
            
        # 夏普比率
        if factor_name == "sharpe_momentum":
            
            # 计算区间平均收益率
            rolling_mean = self.index_exclose.rolling(window).mean() 
            
            # 区间收益率的标准差
            rolling_std = self.index_exclose.rolling(window).std()            
            factor = rolling_mean/rolling_std
            

        # 计算分位数，调整方向
        quantile = self.gen_normal_factor(factor, factor_name, "rolling",window = 250*5)
        
        return quantile
    
    
    # =========================================================================
    # 指标生成：流动性指标
    #
    # [输入]
    # factor_name：指标名称
    # window：区间长度
    # =========================================================================    
    def gen_indus_flow_factor(self, factor_name, window):
        
        # 读取数据
        if factor_name == "volume":
            data = self.index_volume
        if factor_name == "amount":
            data = self.index_amount
        if factor_name == "turn":
            data = self.index_turn
        
        # 计算滚动平均值
        factor = data.rolling(window).mean()
        
        # 计算分位数，调整方向
        quantile = self.gen_normal_factor(factor, factor_name, "rolling", window = 250*5)
        
        return quantile
    
    
    # =========================================================================
    # 指标生成：乖离率指标
    #
    # [输入]
    # factor_name：指标名称
    # window：区间长度
    # =========================================================================  
    def gen_indus_bias_factor(self, factor_name, window):

        # 读取数据               
        if factor_name == "close_bias":
            df = self.indus_close
        if factor_name == "volume_bias":
            df = self.index_volume
        if factor_name == "amount_bias":
            df = self.index_amount
        if factor_name == "turn_bias":
            df = self.index_turn
        if factor_name == 'return_bias':
            df = self.index_exclose
                
        # 计算滚动平均值
        rolling_mean = df.rolling(window).mean()
        
        # 计算数据偏离度
        factor = (df-rolling_mean) / rolling_mean.abs()
            
        # 计算分位数，调整方向
        quantile = self.gen_normal_factor(factor,factor_name,"rolling",window = 250*5)
        
        return quantile
    
    
    # =========================================================================
    # 指标生成：相关系数指标
    #
    # [输入]
    # factor_name：指标名称
    # window：区间长度
    # =========================================================================  
    def gen_roll_corr_factor(self, factor_name, window):
        
        # 读取数据
        if factor_name == "corr_volume_close":
            df = self.index_volume
        if factor_name == "corr_amount_close":
            df = self.index_amount
        if factor_name == "corr_turn_close":
            df = self.index_turn
        
        factor = pd.DataFrame()
        
        # 计算相关系数
        factor = df.rolling(window).corr(self.indus_close)
                       
        # 计算分位数，调整方向
        quantile = self.gen_normal_factor(factor,factor_name,"rolling",window = 250*5)
        
        return quantile
    
    
    # =========================================================================
    # 指标生成：波动率指标
    #
    # [输入]
    # factor_name：指标名称
    # window：区间长度
    # =========================================================================  
    def gen_indus_vol_factor(self, factor_name, window):

        # 日收益率
        indus_return = self.index_exclose   
        
        if factor_name == 'vol':            
            factor = indus_return.rolling(window).std()
            
        if factor_name == 'downvol':    
            
            # 求取下行收益率
            index_down = indus_return < 0
            down_indus_return = indus_return[index_down]
            down_indus_return[down_indus_return.isnull()] = 0   
               
            factor = down_indus_return.rolling(window).std()

        # 计算分位数，调整方向
        quantile = self.gen_normal_factor(factor,factor_name,"rolling",window = 250*5)
        
        return quantile
    
    
    # =========================================================================
    # 指标生成：截面流动性指标
    #
    # [输入]
    # factor_name：指标名称
    # window：区间长度
    # =========================================================================  
    def gen_sec_prop_factor(self, factor_name, window):
        
        # 横截面成交量占比   
        if factor_name == 'section_amount': 
            
            factor = (self.index_amount.T / self.stock_amount.sum(axis=1)).T
            factor = factor.rolling(window).mean()
            
       # 横截面换手率占比 
        if factor_name == 'section_turn': 
            
            factor = (self.index_turn.T / self.stock_turn.sum(axis=1)).T
            factor = factor.rolling(window).mean()
            
        # 计算分位数
        quantile = self.gen_normal_factor(factor, factor_name,"rolling", window = 250*5)
        
        return quantile
    
 
    # =========================================================================
    # 计算指标分位数，转换为拥挤度指标
    #
    # [输入]
    # factor：指标原始数据
    # factor_name：指标名称
    # mehtod：设置为滚动分位数还是历史分位数
    # window：区间长度
    # =========================================================================  
    def gen_normal_factor(self, factor, factor_name, method, window=250*5):
            
        # 数据截取
        factor_sel = factor.dropna(how="all").loc['2004-01-01':,:]

        # 分位数计算代码
        rank_apply = lambda x: np.searchsorted(x,x[-1],sorter=np.argsort(x))/(len(x)-1)     
        
        # 计算历史分位数
        if method == "expanding":
            factor_quantile = factor_sel.expanding(1).apply(rank_apply)
            
        # 计算滚动分位数
        if method == "rolling":
            factor_quantile = factor_sel.rolling(window).apply(rank_apply)        
        
        # 调整方向
        factor_quantile = 1 - factor_quantile if factor_name == 'corr_turn_close' else factor_quantile
        
        # # 上行区间筛选
        # condition = (self.MA_5 > self.MA_10) | (self.MA_10 > self.MA_20) | (self.MA_20 > self.MA_60)
        # factor_quantile[~condition] = -factor_quantile[~condition]
    
        return factor_quantile
    
    
    # =========================================================================
    #  指标计算程序
    # ========================================================================= 
    def cal_data(self, arg):
        
        # 进度提示
        print('指标计算：', arg['指标名称'], arg['窗口长度'])
        
        # 计算指标
        data  = eval('''self.{0}('{1}',{2})'''.format(
            arg['函数名称'], arg['指标名称'], arg['窗口长度']))        

        # 数据存储
        data.to_csv('{}/{}_{}.csv'.format(self.path, arg['指标名称'], arg['窗口长度']), encoding='utf-8-sig')


if __name__ == "__main__":
    
    # 模型初始化
    model = crowd_factors()

    # 指标存储路径
    model.path = "./factor"

    # 生成样本空间
    test_list = pd.DataFrame(
            [['normal_momentum', 'gen_indus_momentum', [5,10,20,40,60]],
              ['turn', 'gen_indus_flow_factor', [5,10,20,40,60]],
              ['turn_bias', 'gen_indus_bias_factor', [20,40,60,120,250]],
              ['return_bias', 'gen_indus_bias_factor', [20,40,60,120,250]],
              ['corr_turn_close', 'gen_roll_corr_factor', [20,40,60]],
              ['vol', 'gen_indus_vol_factor', [5,10,20,40,60]],
              ['section_amount', 'gen_sec_prop_factor', [5,10,20,40,60]],
              ['section_turn', 'gen_sec_prop_factor', [5,10,20,40,60]]],
            columns=['指标名称', '函数名称','窗口长度列表'] )
    
    # 并行遍历计算    
    pool = Pool(5)
    
    for index in test_list.index:
        
        for window in test_list.loc[index, '窗口长度列表']:
            
            # 生成单个计算样本
            sample = pd.Series(
                    [test_list.loc[index, '指标名称'], test_list.loc[index, '函数名称'], window],
                    index=['指标名称', '函数名称','窗口长度'])
            
            # 写入并行库
            pool.apply_async(func=model.cal_data, args=(sample.to_dict(),))
                
    pool.close()
    pool.join()    



