# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 23:54:28 2021

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
import empyrical

plt.style.use("seaborn-white")
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

class model_regress():
    
    # -------------------------------------------------------------------------
    # 实例化，加载基本信息
    # -------------------------------------------------------------------------
    def __init__(self):
                            
        
        # 获取一级行业收盘价
        self.index_close = pd.read_pickle("./data/Wind_indus_close")
                                                            
        # 生成日频交易序列
        self.daily_dates = pd.Series(self.index_close.index, index=self.index_close.index)
        
        # 计算未来20日最大回撤
        daily_ret = self.index_close / self.index_close.shift(1) - 1
        self.indus_max_drawdown = daily_ret.shift(-20).rolling(20).apply(
                    lambda x: empyrical.max_drawdown(x), raw=True)
        
        # 绘图颜色设置
        self.color = np.array([[0.75294118, 0.        , 0.        ],
                               [0.01176471, 0.01176471, 0.01176471],
                               [0.89411765, 0.42745098, 0.04313725],
                               [0.21568627, 0.37647059, 0.57254902],
                               [0.49803922, 0.49803922, 0.49803922]])

                 
    # -------------------------------------------------------------------------
    # 门限回归计算
    # [输入]
    # thres_delta     间隔阈值设定
    # start_time      计算起始时间
    # end_time        计算终止时间
    # factor          拥挤度指标
    # -------------------------------------------------------------------------
    def reg_results(self, thres_delta, start_time, end_time, factor):
        
        threshold = np.arange(0.5, 0.95 + thres_delta, thres_delta)
        output = pd.DataFrame(index=threshold,columns=["回归系数","回归系数显著性","r2"])
        
        # 计算收益率
        y_temp = self.indus_max_drawdown.loc[start_time:end_time,:]        
        x_temp = factor.loc[y_temp.index]
        
        for th in threshold:
            
            # 数据合并为一列
            x_s = x_temp.values[~np.isnan(y_temp)].reshape(-1,1)
            y_s = y_temp.values[~np.isnan(y_temp)].reshape(-1,1)
                        
            # 设定门限值
            y_s = y_s[x_s>=th]
            x_s = x_s[x_s>=th]
            
            if len(x_s) == 0:
                continue 
            
            # # 去掉离群值
            # ub = np.nanmean(y_s)+3*np.nanstd(y_s)
            # lb = np.nanmean(y_s)-3*np.nanstd(y_s)
            # x_s = x_s[(y_s>=lb) & (y_s<=ub)]
            # y_s = y_s[(y_s>=lb) & (y_s<=ub)]
                        
            # 线性回归
            results = sm.OLS(y_s,sm.add_constant(x_s)).fit()
            
            # Newey-West调整
            nw_results = results.get_robustcov_results(cov_type='HAC',maxlags=1)
                        
            try:
                output.loc[th,"回归系数"] = nw_results.params[1]
                output.loc[th,"回归系数显著性"] = nw_results.pvalues[1]
                output.loc[th,"最大回撤中位数"] = np.median(y_s)
            except:
                output.loc[th,"回归系数"] = np.nan
                output.loc[th,"回归系数显著性"] = np.nan
                output.loc[th,"最大回撤中位数"]= np.median(y_s)
        return output
    
    # -------------------------------------------------------------------------
    # 散点图绘制
    # [输入]
    # nparts          分块数
    # start_time      计算起始时间
    # end_time        计算终止时间
    # factor          拥挤度指标
    # -------------------------------------------------------------------------
    def reg_scatter(self, nparts, start_time, end_time, factor):

        # 计算收益率
        y_temp = self.indus_max_drawdown.loc[start_time:end_time,:][factor>=0]
        x_temp = factor.loc[y_temp.index][factor>=0]
        
        # 展开
        x = x_temp.values[~np.isnan(x_temp)].reshape(-1,1)
        y = y_temp.values[~np.isnan(x_temp)].reshape(-1,1)
        
        # scatter
        fig,ax = plt.subplots(figsize=(15,6))
        for i,th in enumerate(np.arange(0,1,1/nparts)):
            cond1 = (x>=th)&(x<th+1/nparts)
            ax.scatter(x[cond1], y[cond1], alpha=0.5, color=self.color[0]+[0,0.1,0.1])
            
    # -------------------------------------------------------------------------
    # 柱形图绘制
    # [输入]
    # output   终版数据
    # -------------------------------------------------------------------------
    def reg_bar(self, output):
        
        fig = plt.figure(figsize=(15,5))
        ax = fig.add_subplot(1,3,1)
        ax.bar(output.index,output["回归系数"], color=self.color[0], width=0.3/output.shape[0])
        ax.tick_params(labelsize=20)
        ax.set_title("回归系数",fontsize=20)

        ax = fig.add_subplot(1,3,2)
        ax.bar(output.index,output["回归系数显著性"], color=self.color[3], width=0.3/output.shape[0])
        ax.tick_params(labelsize=20)
        ax.set_title("回归系数显著性",fontsize=20)

        ax = fig.add_subplot(1,3,3)
        ax.bar(output.index,output["最大回撤中位数"], color=self.color[3], width=0.3/output.shape[0])
        ax.tick_params(labelsize=20)
        ax.set_title("最大回撤中位数",fontsize=20)
         


if __name__ == '__main__':               
    
# # =============================================================================
# #   门限回归 - 单个指标门限回归分析
# # =============================================================================

#     # 模型初始化
#     model = model_regress()

#     start_time = '2010-01-01'
#     end_time = '2021-09-17'
    
#     # 读取指标数据
#     # crowd_factor = pd.read_csv('./factor/'+'turn_5.csv', index_col=0, encoding='utf-8-sig')
#     crowd_factor = pd.read_csv('./factor/'+'corr_turn_close_20.csv', index_col=0, encoding='utf-8-sig')
#     # crowd_factor = pd.read_csv('./factor_up/'+'turn_5.csv', index_col=0, encoding='utf-8-sig')
#     crowd_factor.index = pd.to_datetime(crowd_factor.index)
    
#     # 回测结果
#     results = model.reg_results(0.01, start_time, end_time, crowd_factor)
        
#     # 散点图绘制
#     model.reg_scatter(20, start_time, end_time, crowd_factor)
    
#     # 回归柱状图分析
#     model.reg_bar(results)
#     plt.show()
    
    
# =============================================================================
#   门限回归 - 遍历测试
# =============================================================================

    # 模型初始化
    model = model_regress()

    # start_time = '2010-01-01'
    # end_time = '2021-09-17'

    start_time = '2016-01-01'
    end_time = '2021-09-17'
    
    # 读取指标列表
    factor_name = os.listdir(os.path.join(os.getcwd(),"factor"))

    # 门限回归评价指标计算
    perf = pd.DataFrame(index=["负K值比例","K值相关系数","最大回撤相关系数","显著性占比",
                               "回归系数最大值","回归系数均值","最大回撤最大值","最大回撤均值"])

    for index in range(0, len(factor_name)):
                
        print(index)
        
        # 读取指标数据
        crowd_factor = pd.read_csv('factor/'+ factor_name[index], index_col=0, encoding='utf-8-sig')
        crowd_factor.index = pd.to_datetime(crowd_factor.index)

        # 0.01用来调整门限值的数量
        results = model.reg_results(0.01, start_time, end_time, crowd_factor)
        results["门限值序列"] = results.index
        results = results.astype(float)
        
        # 计算评价指标
        perf[factor_name[index].split('.')[0]] = \
                    [(results.loc[:,"回归系数"]<0).mean(),
                    results["回归系数"].corr(results["门限值序列"], method="pearson"),
                    results["最大回撤中位数"].corr(results["门限值序列"], method="pearson"),                              
                    ((results["回归系数显著性"]<0.01) & (results["回归系数"]<0)).mean(),                                    
                    results.loc[:, "回归系数"].max(),
                    results.loc[:, "回归系数"].mean(),
                    results.loc[:, "最大回撤中位数"].min(),
                    results.loc[:, "最大回撤中位数"].mean()]
            
    # 导出数据
    perf.T.to_excel("门限回归测试结果-2016-2020.xlsx")
    # perf.T.to_excel("门限回归测试结果-2010-2020.xlsx")
