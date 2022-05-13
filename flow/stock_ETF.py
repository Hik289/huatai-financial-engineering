import pandas as pd
import numpy as np
import os
# -----------------------------------------------------------------------------
# ETF指标计算
# -----------------------------------------------------------------------------
class ETF_cap():

    # -------------------------------------------------------------------------
    # 实例化对象，主要用于加载全局数据，避免每次重复加载
    # -------------------------------------------------------------------------
    def __init__(self):
        
        # 获取上级文件路径
        file_path = os.path.abspath(os.path.dirname(os.getcwd()))
        
        # 读取股票列表
        self.stock_info = pd.read_pickle(file_path + '/flow/data/basic/stock_info')    
        
        # 读取A股成交额
        self.stock_amount = pd.read_pickle(file_path + '/flow/data/daily/stock/S_DQ_AMOUNT').loc['2005-01-01':,:] * 1000
        
        # 读取A股流通市值
        self.float_size = pd.read_pickle(file_path + '/flow/data/daily/stock/S_DQ_MV').loc['2005-01-01':,:] * 10000
        
        # 读取A股收盘价
        self.close = pd.read_pickle(file_path + '/flow/data/daily/stock/S_DQ_CLOSE')        
        
        # 指数持仓
        self.index_member = pd.read_pickle(file_path + '/flow/data/daily/stock/AShare_Index_member')
        self.index_member = self.index_member[self.index_member['股票代码'].isin(self.stock_info.index)]
        
        # 所有日频交易日期序列
        self.daily_dates = pd.Series(self.close.index, index=self.close.index)
                    
        # 基金单位净值和福泉精致
        self.fund_close = pd.read_pickle(file_path +"/flow/data/daily/fund/F_NAV_UNIT")
        self.fund_adjclose = pd.read_pickle(file_path +"/flow/data/daily/fund/F_NAV_ADJUSTED")
                
        # 基金份额
        self.float_share = pd.read_pickle(file_path +"/flow/data/daily/fund/ETF_share")
                
        # ETF列表
        self.ETF_info = pd.read_pickle(file_path + "/flow/data/basic/ETF_fund_info")
        self.indus_ETF_list = pd.read_excel(file_path + '/flow/data/basic/ETF基金列表.xlsx', index_col=0)
        self.indus_ETF_list.index = self.indus_ETF_list.index.map(
                lambda x: x.replace('.SH', '.OF').replace('.SZ', '.OF'))
        
        # 读取有交集的基金数据
        fund_list = set(self.indus_ETF_list.index.tolist()) & set(self.float_share.columns.tolist())
        self.indus_ETF_list = self.indus_ETF_list.loc[fund_list,:]

        # 调整收盘价和份额数据的行列索引
        self.float_share = self.float_share.reindex(columns=self.ETF_info.index)
        self.fund_close = self.fund_close.reindex(columns=self.ETF_info.index)
        
        # 数据日期序列重整
        self.float_share = self.float_share.reindex(index=self.daily_dates)
        self.fund_close = self.fund_close.reindex(index=self.daily_dates)


    # -------------------------------------------------------------------------
    # 选定一个ETF代码，统计对应跟踪指数的成分股
    # -------------------------------------------------------------------------
    def get_index_comp(self, idx_name):
        
        # 跟踪指数
        target_index = self.ETF_info.loc[idx_name, "跟踪指数"]
        
        idx_info = self.index_member[self.index_member["指数代码"]==target_index]
        
        if len(idx_info) == 0:            
            return None
        
        else:
            
            # 找到最早的日期和最晚的日期
            all_date_array = np.reshape(idx_info[["纳入日期","剔除日期"]].values, [idx_info.shape[0]*2,1])
            start_date = np.nanmax([pd.to_datetime(np.nanmin(all_date_array)), pd.to_datetime('2005-01-01')])
            end_date = self.daily_dates[-1]
            
            # 生成行列索引
            inds = pd.date_range(start_date,end_date)
            cols = idx_info["股票代码"].drop_duplicates().sort_values()
            
            # 初始化
            idx_comp = pd.DataFrame(False,index=inds,columns=cols,dtype=bool)
            
            # 填充数据
            idx_info = idx_info.fillna(value=end_date)
            
            for i,col in enumerate(idx_info["股票代码"]):
                
                # 获取纳入日期和剔除日期
                start_date = idx_info.iloc[i,2]
                end_date = idx_info.iloc[i,3]
                
                # 区间内数据填充为True
                idx_comp.loc[start_date:end_date,col] = True
            
            # 只选择交易日
            idx_comp = idx_comp.reindex(self.daily_dates).dropna()
            
            return idx_comp
        
    
    # -------------------------------------------------------------------------
    # ETF资金流计算代码
    # -------------------------------------------------------------------------
    def get_ETF_money(self):
                
        # ETF资金流入额
        ETF_money = (self.float_share - self.float_share.shift(1)) *  self.fund_close

        # 空值替换成0
        ETF_money.fillna(0, inplace=True)
        
        # 按日历日提取数据
        ETF_money = ETF_money.resample('D').asfreq() 
        
        # 返回计算结果
        return ETF_money.loc['2014-01-01':,:]
        
    
    # -------------------------------------------------------------------------
    # 个股ETF资金流计算
    # -------------------------------------------------------------------------
    def get_stock_ETF_money(self, ETF_code, ETF_money):
                
        # 统计ETF的成分股
        ETF_comp = self.get_index_comp(ETF_code)
        
        if ETF_comp is None:
            stock_money_flow = pd.DataFrame(0, index=self.daily_dates, columns=self.float_size.columns)
            return stock_money_flow
        
        else:
            # 读取个股流通市值，按照流通市值比例分配ETF资金流数据
            stock_float = self.float_size.loc[ETF_comp.index, ETF_comp.columns].copy()
            
            # 计算ETF成分股比例
            stock_float[ETF_comp==False] = 0
            stock_portion = stock_float.div(stock_float.sum(axis=1), axis=0)
                
            # 计算个股资金流数据
            stock_money_flow = stock_portion.mul(ETF_money.loc[:, ETF_code], axis=0)
            
            # 列表重置
            stock_money_flow = stock_money_flow.reindex(columns=self.float_size.columns)
            
            return stock_money_flow

    
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
    # 增量指标：计算资金流入边际变化情况
    # -------------------------------------------------------------------------
    def gen_money_change(self, origin_data, data_type='normal', freq='W', method='qoq'):
                                     
            
        # 行业北向资金进行归一化
        data_resample = self.data_resample(origin_data, freq, 'sum')
        
        # 归一化
        if data_type == 'float':
                
            # 取流通市值数据，并按照不同频率汇总
            indus_float = self.float_size       
            float_resample = self.data_resample(indus_float, freq, 'sum')
            
            # 将资金数据归一化
            indus_data = data_resample / float_resample
                        
        # 归一化
        if data_type == 'amount':
                
            # 取成交额数据，并按照不同频率汇总
            indus_amount = self.stock_amount      
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
    
    
    # 模型初始化
    model = ETF_cap()
    

# =============================================================================
#   全市场所有ETF
# =============================================================================
    
    # ETF资金流
    ETF_money = model.get_ETF_money().dropna(how='all')
            
    # 汇总资金数据
    money_sum = pd.DataFrame(0, index=model.daily_dates, columns=model.float_size.columns)
    
    # 计算ETF
    for ETF_code in ETF_money.columns:
        
        print(ETF_code)        
        stock_money = model.get_stock_ETF_money(ETF_code, ETF_money)        
        money_sum = money_sum.add(stock_money, fill_value=0)
    
    # 中间过程数据存储
    money_sum.to_pickle(file_path+'/flow/tmp_data/allETF_money')
    money_sum = pd.read_pickle(file_path+'/flow/tmp_data/allETF_money') * 10000

    # 生成调仓信号
    for data_type in ['orig', 'float', 'amount']:     # 是否除以流通市值 
        for freq in ['W','M']:   # 信号发出频率
            for method in ['orig', 'qoq', 'yoy']:  # 信号调整
                
                print(data_type, freq, method)
                
                # 北向流入资金统计
                data_monneyflow = model.gen_money_change(money_sum, data_type, freq, method)
                data_monneyflow.to_pickle(file_path+'/flow/results/allETF_{}_{}_{}'.format(data_type, freq, method))
                
    

# =============================================================================
#   行业和主题ETF
# =============================================================================

    # ETF资金流
    ETF_money = model.get_ETF_money().dropna(how='all')

    # 汇总资金数据
    money_sum = pd.DataFrame(0, index=model.daily_dates, columns=model.float_size.columns)    

    # 计算ETF
    for ETF_code in model.indus_ETF_list.index:
        
        print(ETF_code)        
        stock_money = model.get_stock_ETF_money(ETF_code, ETF_money)        
        money_sum = money_sum.add(stock_money, fill_value=0)
    
    # # 中间过程数据存储
    money_sum.to_pickle(file_path+'/flow/tmp_data/indus_ETF_money')        
    money_sum = pd.read_pickle(file_path+'/flow/tmp_data/indus_ETF_money') * 10000

    # 生成调仓信号
    for data_type in ['orig', 'float', 'amount']:     # 是否除以流通市值 
        for freq in ['W','M']:   # 信号发出频率
            for method in ['orig', 'qoq', 'yoy']:  # 信号调整
                
                print(data_type, freq, method)
                
                # 北向流入资金统计
                data_monneyflow = model.gen_money_change(money_sum, data_type, freq, method)
                data_monneyflow.to_pickle(file_path+'/flow/results/indusETF_{}_{}_{}'.format(data_type, freq, method))
                



# =============================================================================
#   全市场所有ETF
# =============================================================================
    
    # ETF资金流
    ETF_money = model.get_ETF_money().dropna(how='all')
        
    # 只保留同向ETF资金流
    fund_pct_change = model.fund_adjclose - model.fund_adjclose.shift(1)
    ETF_money[((ETF_money>0) & (fund_pct_change<0)) | 
              ((ETF_money<0) & (fund_pct_change>0))] = np.nan
      
    # 汇总资金数据
    money_sum = pd.DataFrame(0, index=model.daily_dates, columns=model.float_size.columns)
    
    # 计算ETF
    for ETF_code in ETF_money.columns:
        
        print(ETF_code)        
        stock_money = model.get_stock_ETF_money(ETF_code, ETF_money)        
        money_sum = money_sum.add(stock_money, fill_value=0)
    
    # 中间过程数据存储
    money_sum.to_pickle(file_path+'/flow/tmp_data/allETFselect_money')
    money_sum = pd.read_pickle(file_path+'/flow/tmp_data/allETFselect_money') * 10000

    # 生成调仓信号
    for data_type in ['orig', 'float', 'amount']:     # 是否除以流通市值 
        for freq in ['W','M']:   # 信号发出频率
            for method in ['orig', 'qoq', 'yoy']:  # 信号调整
                
                print(data_type, freq, method)
                
                # 北向流入资金统计
                data_monneyflow = model.gen_money_change(money_sum, data_type, freq, method)
                data_monneyflow.to_pickle(file_path+'/flow/results/allETFselect_{}_{}_{}'.format(data_type, freq, method))
                


# =============================================================================
#   行业和主题ETF - 同向时资金流
# =============================================================================

    # ETF资金流
    ETF_money = model.get_ETF_money().dropna(how='all')
        
    # 只保留同向ETF资金流
    fund_pct_change = model.fund_adjclose - model.fund_adjclose.shift(1)
    ETF_money[((ETF_money>0) & (fund_pct_change<0)) | 
              ((ETF_money<0) & (fund_pct_change>0))] = np.nan
        
    # 汇总资金数据
    money_sum = pd.DataFrame(0, index=model.daily_dates, columns=model.float_size.columns)    

    # 计算ETF
    for ETF_code in model.indus_ETF_list.index:
        
        print(ETF_code)        
        stock_money = model.get_stock_ETF_money(ETF_code, ETF_money)        
        money_sum = money_sum.add(stock_money, fill_value=0)
        
    # # 中间过程数据存储
    money_sum.to_pickle(file_path+'/flow/tmp_data/indusETFselect_money')        
    money_sum = pd.read_pickle(file_path+'/flow/tmp_data/indusETFselect_money') * 10000

    # 生成调仓信号
    for data_type in ['orig', 'float', 'amount']:     # 是否除以流通市值 
        for freq in ['W','M']:   # 信号发出频率
            for method in ['orig', 'qoq', 'yoy']:  # 信号调整
                
                print(data_type, freq, method)
                
                # 北向流入资金统计
                data_monneyflow = model.gen_money_change(money_sum, data_type, freq, method)
                data_monneyflow.to_pickle(file_path+'/flow/results/indusETFselect_{}_{}_{}'.format(data_type, freq, method))

    