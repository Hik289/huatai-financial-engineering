from utils import BacktestUtils, PerfUtils, DataUtils
from utils.BaseClass import BaseClass
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号


class model_backtest(BaseClass):

    
    # -------------------------------------------------------------------------
    # 实例化，加载基本信息
    # -------------------------------------------------------------------------
    def __init__(self):
            
        # 初始化父对象
        super().__init__()
        
        # 获取行业收盘价
        self.indus_close = DataUtils.get_daily_info('indus', 'close')
        
        # 生成日频交易序列
        self.daily_dates = pd.Series(self.indus_close.index, index=self.indus_close.index)
        
        # 读取沪深300指数收盘价
        self.hs300_close = pd.read_excel('data/中证800.xlsx', index_col=0)
        
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
    # base           参考基准
    # fee            手续费
    # -------------------------------------------------------------------------
    def backtest(self, df_factor_input, start_date, end_date, base, fee=0):
        
        df_factor = df_factor_input.copy()
        
        # 当行业景气度指标给定的最后一个日期大于最后一个交易日时（月末容易出现）
        # 最后一个交易信号无法调仓
        if df_factor.index[-1] >= self.daily_dates[-1]:
            df_factor = df_factor.drop(index=df_factor.index[-1])
            
        # 调仓日期为生成信号的下一天，即月初第一个交易日        
        # df_factor.index = [self.daily_dates.iloc[self.daily_dates.index.tolist().index(i) + 1
        #                                          ] for i in df_factor.index]
        df_factor.index = [self.daily_dates[self.daily_dates > i].index[0] for i in df_factor.index]        
        
        df_factor=df_factor[~df_factor.index.duplicated(keep='last')]
        
        # 根据输入的行业指标，计算多头和空头持仓
        long_portion_all = df_factor.copy()*0  
        short_portion_all = df_factor.copy()*0 
        # pd.DataFrame(index=df_factor.index,columns=df_factor.columns,data=1/len(df_factor.columns))
        
        # 遍历所有日期
        for date in df_factor.index:    
            
            # 一般情况，标为1的行业为多头持仓，这里简化为指标值大于零的行业为多头持仓
            if sum(df_factor.loc[date,:] > 0) != 0:
                long_portion_all.loc[date,df_factor.loc[date,:]>0] = 1/sum(df_factor.loc[date,:] > 0)
            elif sum(df_factor.loc[date,:] > 0) ==0:
                long_portion_all.loc[date]=np.full(shape=len(df_factor.columns),fill_value=1/len(df_factor.columns))
                
            # 一般情况，标为-1的行业为空头持仓，这里简化为指标值小于零的行业为空头持仓
            if sum(df_factor.loc[date,:] < 0) != 0:
                short_portion_all.loc[date,df_factor.loc[date,:]<0] = 1/sum(df_factor.loc[date,:] < 0)
            elif sum(df_factor.loc[date,:] < 0) ==0:
                short_portion_all.loc[date]=np.full(shape=len(df_factor.columns),fill_value=1/len(df_factor.columns))
                
        # 对持仓进行时间截断
        long_portion = long_portion_all[start_date:end_date]
        short_portion = short_portion_all[start_date:end_date]
        
        # 参照基准 - 行业收益率等权
        if base == 'mean' :   
            indus_return = self.indus_close.pct_change()
            base_return = indus_return.mean(axis=1) + 1
            base_close = base_return.cumprod()
            base_nav = base_close.loc[start_date:end_date]
            
        # 参照基准 - 沪深300
        elif base == 'hs300':
            base_nav = self.hs300_close.loc[start_date:end_date,'close']
            
        # 计算绝对净值        
        nav = pd.DataFrame(columns=['多头策略','空头策略','基准'])
        
        # 回测,计算策略净值
        nav['多头策略'], df_indus_return = BacktestUtils.cal_nav(long_portion, self.indus_close[start_date:end_date], base_nav, fee)
            
        nav['空头策略'], df_indus_return_short = BacktestUtils.cal_nav(short_portion, self.indus_close[start_date:end_date], base_nav, fee)
        
        # 基准净值归一化
        nav['基准'] = base_nav / base_nav.values[0]
          
        # 计算相对净值
        nav_relative = pd.DataFrame(columns=['多头/基准','空头/基准'])
        nav_relative['多头/基准'] = nav['多头策略'] / nav['基准'] 
        nav_relative['空头/基准'] = nav['空头策略'] / nav['基准']
        
        # 返回绝对净值曲线，相对净值曲线，多头持仓和空头持仓
        return nav, nav_relative, long_portion, short_portion, df_indus_return


    # -------------------------------------------------------------------------
    # 计算回测指标
    # [输入]
    # nav             净值
    # refresh_dates   调仓日期
    # -------------------------------------------------------------------------
    def performance(self, nav, refresh_dates):

        # 初始化结果矩阵
        perf = pd.DataFrame(index=['多头', '空头', '基准'], 
                            columns=['年化收益率','年化波动率','夏普比率','最大回撤',
                                     '年化超额收益率','超额收益年化波动率','信息比率',
                                     '超额收益最大回撤','调仓胜率'])
                
        # 计算多头收益
        long_perf = PerfUtils.excess_statis(nav['多头策略'], nav['基准'], refresh_dates)
        perf.loc['多头',:] = long_perf.loc['策略',:]
        perf.loc['基准',:] = long_perf.loc['基准',:]
        
        # 计算空头收益
        short_perf = PerfUtils.excess_statis(nav['空头策略'], nav['基准'], refresh_dates)
        perf.loc['空头',:] = short_perf.loc['策略',:]
        
        # 计算多头相对空头胜率
        ls_perf = PerfUtils.excess_statis(nav['多头策略'], nav['空头策略'], refresh_dates)
        perf.loc['多头','多空胜率'] = ls_perf.loc['策略','调仓胜率']
        
        return perf
    
    
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
    
    # -------------------------------------------------------------------------
    # 将给定路径下的景气度指标进行叠加
    # [输入]
    # path        需要进行复合的单项行业景气度指标存储路径
    # factors     进行复合的行业指标名称
    # cut_index   输入对于指标进行截断的时间点
    # -------------------------------------------------------------------------
    def cal_merge_factor(self, path, factors, cut_index):
        
        # 复合指标记录变量初始化
        m_factor = None
        
        # 遍历给定的单项行业景气度指标   
        for factor_name in factors:
            
            # 读取单项行业景气度指标
            df = pd.read_pickle(path + factor_name)
            
            # 将不同单项景气度指标进行叠加
            if m_factor is None:
                m_factor = df
                
            else:
                # 不同单项景气度指标时间起点有区别
                m_index = m_factor.index if len(df.index) > len(m_factor.index) else df.index
                m_factor.loc[m_index.intersection(m_factor.index),:] = m_factor.loc[m_index.intersection(m_factor.index),:] + df.loc[m_index.intersection(df.index),:]    
        
        return m_factor.loc[cut_index:,:]
    
    # -------------------------------------------------------------------------
    # 依据相关系数筛选指标
    # [输入]
    # factor_result  单因子回测结果
    # return_all     单因子净值曲线结果
    # threshold      筛选因子时对于因子年化超额收益的要求
    # -------------------------------------------------------------------------
    def factor_corr_select(self,factor_result,return_all,threshold=0.01):
        
        # 初始化指标基础池
        factor_select = factor_result[factor_result['年化超额收益率']>threshold].sort_values(by='年化收益率', ascending=False)
        
        # 筛选指标
        final_factors = [factor_select.index[0]]
        
        # 循环遍历纳入指标
        for index in factor_select.index[1:]:
        
            # 最新指标
            assume_factors = final_factors + [index]
        
            # 计算相关系数
            corr = return_all.loc[:, assume_factors].corr()

            # 获取结果
            if corr.loc[final_factors, index].max() > 0.5:
                continue
            else:
                final_factors = assume_factors
    
        final_factor_details = factor_select.loc[final_factors,:]
    
        return final_factor_details
    
    # -------------------------------------------------------------------------
    # 逐个添加指标对于复合指标进行测试
    # [输入]
    # final_factors  筛选出的因子
    # file_name      文件存储名称
    # ------------------------------------------------------------------------- 
    def cal_merge_perf(self,path,final_factors,file_name):
    
        # 指标业绩汇总
        merge_perf = pd.DataFrame(index = np.arange(0, len(final_factors)),
                     columns = ['年化收益率', '年化波动率', '夏普比率', '最大回撤', 
                               '年化超额收益率', '超额收益年化波动率', '信息比率',
                                '超额收益最大回撤', '调仓胜率', '多空胜率'])
        
        for factor_i in range(0, len(final_factors)):
            
            # 指标复合
            merge_factor = model.cal_merge_factor(path, final_factors[:factor_i+1], cut_index='2006-01-01')
        
            # 依据复合景气度指标的计算多空行业
            merge_ls_factor = model.cal_merge_ls_factor(merge_factor, indus_num=5)
    
            # 回测过程
            [nav, nav_relative, long_portion, short_portion, df_indus_return] = \
                     model.backtest(merge_ls_factor, '2010-01-01', '2021-05-30', base='mean')
                  
            # 计算回测业绩指标
            perf = model.performance(nav, long_portion.index)
    
            # 各指标表现汇总
            merge_perf.loc[factor_i,:] = perf.loc['多头',:].values
        
        # 存储回测结果
        merge_perf.to_excel('results/'+file_name+'.xlsx')
        
        return merge_perf 
   
    # -------------------------------------------------------------------------
    # 对于指标进行截面的归一化，将指标数值限制在[-1,1]之间
    # [输入]
    # df  进行归一化的指标
    # -------------------------------------------------------------------------  
    def normalization(self,df):
        for i in df.index:
            
            # 获取截面数据绝对值的最大值
            max_val=np.abs(df.loc[i]).max()
            if max_val!=0:
                df.loc[i]/=max_val

        return df
    # -------------------------------------------------------------------------
    # 对于财务数据、分析师预期数据按照权重进行复合，对于复合指标进行回测
    # [输入]
    # est_factors  分析师预期数据
    # fin_factors  财务数据
    # weight       财务数据在财报期的权重
    # path         路径
    # -------------------------------------------------------------------------                    
    def plot_merge_perf(self,path,est_factors,fin_factors,weight=1):
        
        # 指标复合
        merge_factor=pd.DataFrame()
         
        # 只包含财务数据
        if est_factors==[]:
            merge_factor=model.cal_merge_factor(path, fin_factors, cut_index='2006-01-01') 
        else:
            # 只包含分析师预期数据
            if fin_factors==[]:
                merge_factor=model.cal_merge_factor(path, est_factors, cut_index='2006-01-01')
            else:
                fin_df=model.cal_merge_factor(path, fin_factors, cut_index='2006-01-01')
                est_df=model.cal_merge_factor(path, est_factors, cut_index='2006-01-01')
                
                # 获取财报期的信号日
                signal_df=pd.read_pickle(path + fin_factors[0])
                signal_df=signal_df.replace(0,np.nan)
                financial_signal_date=signal_df.index[~signal_df.isnull().all(1).values]
                
                # 进行截面的归一化
                fin_df=model.normalization(fin_df)
                est_df=model.normalization(est_df)
                
                merge_factor=pd.DataFrame(index=fin_df.index.intersection(est_df.index),columns=fin_df.columns,data=0)
                
                # 将财务数据、分析师预期数据进行复合
                for i in merge_factor.index:
                    if i in financial_signal_date:
                        merge_factor.loc[i]=weight*fin_df.loc[i]+(1-weight)*est_df.loc[i]
                    else:
                        merge_factor.loc[i]=est_df.loc[i]

        # 依据复合景气度指标的计算多空行业
        merge_ls_factor = model.cal_merge_ls_factor(merge_factor, indus_num=5)
    
        # 回测过程
        [nav, nav_relative, long_portion, short_portion, df_indus_return] = model.backtest(merge_ls_factor, '2010-01-01', '2021-05-30', base='mean', fee=0.000)
        
        # 计算回测业绩指标
        perf = model.performance(nav, long_portion.index)
        
        # 作图
        nav.plot()
        
        return perf
        
        
if __name__ == '__main__':  

# =============================================================================
#   单个指标测试
# =============================================================================
    
    # 模型初始化
    model = model_backtest()
    
    # 路径
    path='gen_factor/result/' 
    
    # 读取指标
    df_factor = pd.read_pickle(path + 'net_profit_excl')
    
    # 回测过程
    [nav, nav_relative, long_portion, short_portion, df_indus_return] = model.backtest(df_factor, '2010-01-01', '2021-05-30', base='mean')
              
    # 计算回测业绩指标
    perf = model.performance(nav, long_portion.index)
    
    # 作图
    # nav.plot()
    # nav_relative.plot()
    
# =============================================================================
#   遍历测试
# =============================================================================
    
    # 回测起止时间
    start_date='2010-01-01'
    end_date= '2021-05-30'
    
    for root, dirs, factors in os.walk(path):
        pass
    
    data = []
    for factor_name in factors:
        data.append(factor_name.split('_')[-1])
    
    # 所有信号日期及行业
    total_signal_dates=pd.date_range(start_date, end_date, freq='W')
    total_indus=DataUtils.get_daily_info('indus', 'close').columns
    
    # 指标业绩汇总
    merge_perf = pd.DataFrame(index = factors,columns = ['年化收益率', '年化波动率', '夏普比率', '最大回撤', '年化超额收益率', '超额收益年化波动率', '信息比率','超额收益最大回撤', '调仓胜率'])
    
    # 净值曲线汇总
    nav_all = pd.DataFrame(columns = factors)
    
    # 所有分析师预期因子及所有财务因子
    est_list=[]
    fin_list=[]
    
    for factor_name in factors:
        
        # 读取指标
        df_factor = pd.read_pickle(path + factor_name)
        df_factor = df_factor.astype('float') 
        df_factor=df_factor.loc[start_date:end_date]
        
        # 分别提取分析师预期因子与财务因子
        if factor_name[0:3]!='est':
            fin_list.append(factor_name)
        else:
            est_list.append(factor_name)
        
        # 回测过程
        [nav, nav_relative, long_portion, short_portion, df_indus_return] = model.backtest(df_factor, start_date, end_date, base='mean')
        
        # 计算回测业绩指标
        perf = PerfUtils.excess_statis(nav['多头策略'], nav['基准'], long_portion.index)
    
         # 各指标表现汇总
        merge_perf.loc[factor_name,:] = perf.loc['策略',:].values
        nav_all.loc[:,factor_name] = nav['多头策略']
    
        # 作图
        # nav.plot(title=factor_name)
    
    # 因子名称储存
    factor_list=est_list+fin_list
    
    # 存储基准收益率
    nav_all.loc[:, '基准'] = nav['基准']
    
    # 存储回测结果
    merge_perf.to_excel('results/回测结果汇总.xlsx')
    
     # 存储净值曲线
    nav_all.to_excel('results/净值曲线汇总.xlsx')
    
# =============================================================================
#   因子筛选
# =============================================================================
    # 回测结果和指标名称匹配
    factor_names = pd.read_excel('results/指标名称.xlsx', index_col=0)
    factor_names = factor_names.set_index('指标代码')
    merge_perf=pd.read_excel('results/回测结果汇总.xlsx',index_col=0)
    factor_result = pd.concat([factor_names, merge_perf],axis=1)
    factor_result=factor_result.loc[factor_list]
    
    # 存储净值曲线
    nav_all = pd.read_excel('results/净值曲线汇总.xlsx', index_col=0)
    return_all = nav_all.pct_change().sub(nav_all.loc[:, '基准'].pct_change(), axis=0)
    
    # 通过相关系数进行因子筛选
    est_factor_details=model.factor_corr_select(factor_result[factor_result.index.str.contains('est_')],return_all,0.01)
    
    fin_factor_details=model.factor_corr_select(factor_result[~factor_result.index.str.contains('est_')],return_all,0.005)
    
# =============================================================================
#   财务、分析师预期数据分别复合指标测试
# =============================================================================
    est_merge_perf=model.cal_merge_perf(path,est_factor_details.index,'复合预期指标回测结果')
    
    fin_merge_perf=model.cal_merge_perf(path,fin_factor_details.index,'复合财务指标回测结果')

    perf_est=model.plot_merge_perf(path,est_factor_details.index.tolist(),[],0)
    
    perf_fin=model.plot_merge_perf(path,[],fin_factor_details.index.tolist(),0)
    
# =============================================================================
#   财务、分析师预期数据复合指标测试
# =============================================================================

    perf_merge=model.plot_merge_perf(path,est_factor_details.index.tolist(),fin_factor_details.index.tolist(),0.5)