import sys
sys.path.append("..")
import utils.DataUtils as DataUtils
from utils.BaseClass import BaseClass
import pandas as pd
import numpy as np
import datetime

# -----------------------------------------------------------------------------
# 个股一致预期指标
# -----------------------------------------------------------------------------
class model_est(BaseClass):

    # -------------------------------------------------------------------------
    # 实例化对象，主要用于加载全局数据，避免每次重复加载
    # -------------------------------------------------------------------------
    def __init__(self, data_name):

        # 初始化父对象
        super().__init__()

        # 读取数据，除FY1FY2数据以外不用做特殊处理
        if data_name.find('FY1FY2') == -1:
            self.est_data = DataUtils.get_daily_info('stock', data_name)
        else:
            # 读取FY1数据
            self.est_data = DataUtils.get_daily_info('stock', data_name[:-3])
            
            # 读取FY2数据
            self.est_data_FY2 = DataUtils.get_daily_info('stock', data_name[:-6] + 'FY2')
            data_FY2_index = self.est_data_FY2.index.tolist()

            # 将FY1数据2月和3月替换成FY2
            for i, ii in self.est_data.iterrows():
                
                # 有时候FY1、FY2的日期对不齐，加了个判断条件
                if i.month in [2, 3] and i in data_FY2_index:
                    self.est_data.loc[i] = self.est_data_FY2.loc[i]

        # 生成日频时间索引
        daily_dates=pd.date_range('2007-01-01',self.est_data.index[-1],freq='D')
        
        # 梳理股票一致预期数据                    
        self.est_data = self.est_data.loc[:, self.est_data.columns.intersection(self.stock_id)]
        
        # 读取个股相应的市值数据
        self.S_DQ_MV = DataUtils.get_daily_info('stock', 'S_DQ_MV')
        self.S_DQ_MV = self.S_DQ_MV.loc[:, self.S_DQ_MV.columns.intersection(self.stock_id)]

        self.est_data = self.est_data.reindex(daily_dates,method='ffill')
        self.S_DQ_MV = self.S_DQ_MV.reindex(daily_dates,method='ffill')


    # -------------------------------------------------------------------------
    # 获取每周周日日期
    # [输入]
    # start_date     开始时间
    # end_date       终止时间
    # frequency      周频
    # -------------------------------------------------------------------------
    def gen_panel_dates(self, start_date, end_date, frequency='W'):

        # 指标时间区间
        month_end_dates = pd.date_range(start=start_date, end=end_date, freq=frequency)

        return month_end_dates

    # -------------------------------------------------------------------------
    # 根据汇总的截面数据，按行业统计一致数据，相加方式为直接相加
    # [输入]
    # df_cur_panel   当前时间截面数据
    # df_prev_panel  前一时间截面数据
    # -------------------------------------------------------------------------
    def merge_panel_value(self, df_cur_panel, df_prev_panel):

        # 按行业分组进行统计
        indus_group = df_prev_panel[['est_data']].groupby(df_prev_panel['indus_belong'])

        # 按组统计,统计个数
        indus_group_statis = indus_group.count()

        # 行业内只有小于五只个股有数据时置为空值
        indus_group_statis[indus_group_statis < 5] = np.nan

        # 生成统计个股业绩增长DataFrame
        df_diff_panel = df_cur_panel.copy()

        # 计算变化量
        df_diff_panel[['est_data']] = df_cur_panel[['est_data']] - df_prev_panel[['est_data']]

        # 按行业分组进行统计
        indus_group_diff = df_diff_panel[['est_data']].groupby(df_prev_panel['indus_belong'])

        # 按组统计一致预期数据增长个股的指标差值之和
        indus_group_diff_statis = indus_group_diff.sum()

        # 一致预期数据个股的指标差值之和
        return indus_group_diff_statis['est_data']

    # -------------------------------------------------------------------------
    # 计算个股合成一致预期数据变动值
    # [输入]
    # panel_dates    调仓日期序列
    # diff_method    差分方法: 同比-'yoy'/环比-'qoq'
    # -------------------------------------------------------------------------
    def gen_diff_value(self, panel_dates):

        # 初始化返回值
        df_diff_data = pd.DataFrame(index=panel_dates, columns=self.indus_name)
            
        # 获取去年同期上月末的值
        weekly_end=pd.DataFrame(index=panel_dates,columns=['prev'])
        monthly_end=weekly_end.resample('M').last()
        monthly_end['prev']=monthly_end.index.shift(-13)
        
        prev_date_list=[monthly_end.loc[i.strftime('%Y-%m'),'prev'][0].strftime('%Y-%m-%d') for i in weekly_end.index]
        weekly_end['prev']=prev_date_list
        
        # 遍历每个截面
        for date in panel_dates:
            # 获取前期日期
            prev_date = weekly_end.loc[date,'prev']
            
            # 对齐操作一：保留行业归属不变的股票
            cur_indus_belong = self.indus_belong.loc[date, :]
            prev_indus_belong = self.indus_belong.loc[prev_date, :]
            valid_stocks = list(cur_indus_belong[cur_indus_belong == prev_indus_belong].index)

            # 获取当前截面股票数据
            df_cur = pd.DataFrame(
                {'est_data': self.est_data.loc[date, valid_stocks],
                 'indus_belong': self.indus_belong.loc[date, valid_stocks]},
                index=valid_stocks)
            df_cur = df_cur.dropna()

            # 获取前期截面股票数据
            df_prev = pd.DataFrame(
                {'est_data': self.est_data.loc[prev_date, valid_stocks],
                 'indus_belong': self.indus_belong.loc[prev_date, valid_stocks]},
                index=valid_stocks)
            df_prev = df_prev.dropna()

            # 对齐操作二：前后期去除空值后，可能仍然有不完全匹配的地方
            choosed_stock = list(set(df_cur.index) & set(df_prev.index))

            # 计算当前截面汇总值
            df_diff_data.loc[date, :] = self.merge_panel_value(
                df_cur.loc[choosed_stock, :], df_prev.loc[choosed_stock, :])
            
        # 数据类型转换为数值型
        df_diff_data = df_diff_data.astype('float')

        return df_diff_data

    # -------------------------------------------------------------------------
    # 行业指标计算程序
    # [输入]
    # panel_dates    调仓日期序列
    # diff_method    差分方法: 同比-'yoy'/环比-'qoq'
    # -------------------------------------------------------------------------
    def gen_factor_value(self, panel_dates):

        # 计算行内个股一致预期数据提升比例
        df_diff_data = model.gen_diff_value(panel_dates)

        # 创建行业指标空矩阵
        factor = pd.DataFrame(np.zeros_like(df_diff_data.values),
                              index=df_diff_data.index, columns=df_diff_data.columns)

        # 差值大于0为正
        factor[df_diff_data > 0] = 1
        factor[df_diff_data < 0] = -1

        return factor.loc[panel_dates, :]

    # -------------------------------------------------------------------------
    # 以市值加权方式计算比值类指标增量
    # [输入]
    # df_cur_panel   当前时间截面数据
    # df_prev_panel  前一时间截面数据
    # -------------------------------------------------------------------------
    def merge_panel_ratio(self, df_cur_panel, df_prev_panel):

        # 按行业分组进行统计
        indus_group = df_prev_panel[['est_data']].groupby(df_prev_panel['indus_belong'])

        # 按组统计,统计个数
        indus_group_statis = indus_group.count()

        # 按流通市值加权
        df_cur_panel.loc[:, 'est_data'] = df_cur_panel.loc[:, 'est_data'] * df_cur_panel.loc[:, 'size']
        df_prev_panel.loc[:, 'est_data'] = df_prev_panel.loc[:, 'est_data'] * df_prev_panel.loc[:, 'size']

        # 计算变化量
        indus_group = df_cur_panel[['est_data']].groupby(df_cur_panel['indus_belong']).sum()['est_data'] / \
                      df_cur_panel[['size']].groupby(df_cur_panel['indus_belong']).sum()['size'] - \
                      df_prev_panel[['est_data']].groupby(df_prev_panel['indus_belong']).sum()['est_data'] / \
                      df_prev_panel[['size']].groupby(df_prev_panel['indus_belong']).sum()['size']

        # 行业内只有小于五只个股有数据时置为空值
        indus_group[indus_group_statis['est_data'] < 5] = np.nan

        # 一致预期数据个股的指标差值之和
        return indus_group


    # -------------------------------------------------------------------------
    # 行业内个股一致预期数据提升比例
    # [输入]
    # panel_dates    调仓日期序列
    # diff_method    差分方法: 同比-'yoy'/环比-'qoq'
    # -------------------------------------------------------------------------
    def gen_diff_ratio(self, panel_dates):

        # 初始化返回值
        df_diff_data = pd.DataFrame(index=panel_dates, columns=self.indus_name)
            
        # 获取去年同期上月末的值
        weekly_end=pd.DataFrame(index=panel_dates,columns=['prev'])
        monthly_end=weekly_end.resample('M').last()
        monthly_end['prev']=monthly_end.index.shift(-13)
        
        prev_date_list=[monthly_end.loc[i.strftime('%Y-%m'),'prev'][0].strftime('%Y-%m-%d') for i in weekly_end.index]
        weekly_end['prev']=prev_date_list
        
        # 遍历每个截面
        for date in panel_dates:
            # 获取前期日期
            prev_date = weekly_end.loc[date,'prev']

            # 对齐操作一：保留行业归属不变的股票
            cur_indus_belong = self.indus_belong.loc[date, :]
            prev_indus_belong = self.indus_belong.loc[prev_date, :]
            valid_stocks = list(cur_indus_belong[cur_indus_belong == prev_indus_belong].index)

            # 获取当前截面股票数据
            df_cur = pd.DataFrame(
                {'est_data': self.est_data.loc[date, valid_stocks],
                 'indus_belong': self.indus_belong.loc[date, valid_stocks],
                 'size': self.S_DQ_MV.loc[date, valid_stocks]},
                index=valid_stocks)
            df_cur = df_cur.dropna()

            # 获取前期截面股票数据
            df_prev = pd.DataFrame(
                {'est_data': self.est_data.loc[prev_date, valid_stocks],
                 'indus_belong': self.indus_belong.loc[prev_date, valid_stocks],
                 'size': self.S_DQ_MV.loc[prev_date, valid_stocks]},
                index=valid_stocks)
            df_prev = df_prev.dropna()

            # 对齐操作二：前后期去除空值后，可能仍然有不完全匹配的地方
            choosed_stock = list(set(df_cur.index) & set(df_prev.index))
     
            
            # 计算当前截面汇总值
            df_diff_data.loc[date, :] = self.merge_panel_ratio(
                df_cur.loc[choosed_stock, :], df_prev.loc[choosed_stock, :])

        # 数据类型转换为数值型
        df_diff_data = df_diff_data.astype('float')

        return df_diff_data

    # -------------------------------------------------------------------------
    # 行业指标计算程序
    # [输入]
    # panel_dates    调仓日期序列
    # diff_method    差分方法: 同比-'yoy'/环比-'qoq'
    # -------------------------------------------------------------------------
    def gen_factor_ratio(self, panel_dates):

        # 计算行内个股一致预期数据提升比例
        df_diff_data = model.gen_diff_ratio(panel_dates)

        # 创建行业指标空矩阵
        factor = pd.DataFrame(np.zeros_like(df_diff_data.values),
                              index=df_diff_data.index, columns=df_diff_data.columns)

        # 差值大于0为1
        factor[df_diff_data > 0] = 1
        factor[df_diff_data < 0] = -1

        return factor
    
    
if __name__ == '__main__':

    # 数据类型
    types = ['FY2','CAGR']
    
    # 数据预处理
    for data_type in types:

        # 数据类型
        factor_dict = {'EST_EPS_' + data_type: 'EST_EPS',
                        'EST_ROE_' + data_type: 'EST_ROE',
                        'NET_PROFIT_' + data_type: 'NET_PROFIT',
                        'EST_OPER_REVENUE_' + data_type: 'EST_OPER_REVENUE',
                        'EST_CFPS_' + data_type: 'EST_CFPS',
                        'EST_DPS_' + data_type: 'EST_DPS',
                        'EST_BPS_' + data_type: 'EST_BPS',
                        'EST_EBIT_' + data_type: 'EST_EBIT',
                        'EST_EBITDA_' + data_type: 'EST_EBITDA',
                        'EST_TOTAL_PROFIT_' + data_type: 'EST_TOTAL_PROFIT',
                        'EST_OPER_PROFIT_' + data_type: 'EST_OPER_PROFIT'}

        for data_name, value in factor_dict.items():

            try:
                # 模型初始化
                model = model_est(data_name)

                # 生成每月最后一个交易日序列作为日期索引
                panel_dates = model.gen_panel_dates('2009-01-01', '2021-05-31', 'W')

                # 比值类的行业指标通过流通市值加权，数值类的指标直接相加
                if value in ['EST_EPS', 'EST_ROE', 'EST_CFPS','EST_DPS','EST_BPS']:

                    # 计算行业指标 - 比值类型
                    df_factor = model.gen_factor_ratio(panel_dates)

                else:
                    # 计算行业指标 - 数值类型
                    df_factor = model.gen_factor_value(panel_dates)

                # 行业指标存储
                df_factor.to_pickle('gen_factor/result/est_stock_{}'.format(data_name))

                print('指标计算成功：', data_type, data_name)
                
            except:
                
                print('指标计算失败：', data_type, data_name)