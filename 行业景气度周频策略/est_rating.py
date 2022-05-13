import sys
sys.path.append("..")
import utils.DataUtils as DataUtils
from utils.BaseClass import BaseClass
import pandas as pd
import numpy as np


# -----------------------------------------------------------------------------
# 一致预期关注度类指标
# -----------------------------------------------------------------------------
class model_rating(BaseClass):

    # -------------------------------------------------------------------------
    # 实例化对象，主要用于加载全局数据，避免每次重复加载
    # -------------------------------------------------------------------------
    def __init__(self, factor_name, FY):

        # 初始化父对象
        super().__init__()

        # 一致预期综合评级同比  WRATING_AVG
        if factor_name == 'WRATING_AVG':
                
            self.basic_data = DataUtils.get_daily_info('stock', 'S_WRATING_AVG')
            self.basic_data = self.basic_data.loc[:, self.basic_data.columns.intersection(self.stock_id)]
            
        # 评级调高家数占比同比  WRATING_UPGRADE
        if factor_name == 'WRATING_UPGRADE':
            
            S_WRATING_UPGRADE = DataUtils.get_daily_info('stock', 'S_WRATING_UPGRADE')
            S_WRATING_DOWNGRADE = DataUtils.get_daily_info('stock', 'S_WRATING_DOWNGRADE')
            S_WRATING_MAINTAIN = DataUtils.get_daily_info('stock', 'S_WRATING_MAINTAIN')
            basic_data = S_WRATING_UPGRADE / (S_WRATING_UPGRADE + S_WRATING_DOWNGRADE + S_WRATING_MAINTAIN)
            self.basic_data = basic_data.loc[:,basic_data.columns.intersection(self.stock_id)]
            
        # 买入评级家数占比同比  WRATING_NUMOFBUY
        if factor_name == 'WRATING_NUMOFBUY':
            
            S_WRATING_NUMOFBUY = DataUtils.get_daily_info('stock', 'S_WRATING_NUMOFBUY')
            S_WRATING_NUMOFOUTPERFORM = DataUtils.get_daily_info('stock', 'S_WRATING_NUMOFOUTPERFORM')
            S_WRATING_NUMOFHOLD = DataUtils.get_daily_info('stock', 'S_WRATING_NUMOFHOLD')
            S_WRATING_NUMOFUNDERPERFORM = DataUtils.get_daily_info('stock', 'S_WRATING_NUMOFUNDERPERFORM')
            S_WRATING_NUMOFSELL = DataUtils.get_daily_info('stock', 'S_WRATING_NUMOFSELL')
            
            basic_data = S_WRATING_NUMOFBUY / (S_WRATING_NUMOFBUY + S_WRATING_NUMOFOUTPERFORM + \
                       S_WRATING_NUMOFHOLD + S_WRATING_NUMOFUNDERPERFORM + S_WRATING_NUMOFSELL)
            self.basic_data = basic_data.loc[:,basic_data.columns.intersection(self.stock_id)]
            
        # 增持以上评级家数占比同比  WRATING_NUMOFOUTPERFORM
        if factor_name == 'WRATING_NUMOFOUTPERFORM':
            
            S_WRATING_NUMOFBUY = DataUtils.get_daily_info('stock', 'S_WRATING_NUMOFBUY')
            S_WRATING_NUMOFOUTPERFORM = DataUtils.get_daily_info('stock', 'S_WRATING_NUMOFOUTPERFORM')
            S_WRATING_NUMOFHOLD = DataUtils.get_daily_info('stock', 'S_WRATING_NUMOFHOLD')
            S_WRATING_NUMOFUNDERPERFORM = DataUtils.get_daily_info('stock', 'S_WRATING_NUMOFUNDERPERFORM')
            S_WRATING_NUMOFSELL = DataUtils.get_daily_info('stock', 'S_WRATING_NUMOFSELL')
            
            basic_data = (S_WRATING_NUMOFBUY + S_WRATING_NUMOFOUTPERFORM) / \
                        (S_WRATING_NUMOFBUY + S_WRATING_NUMOFOUTPERFORM + \
                         S_WRATING_NUMOFHOLD + S_WRATING_NUMOFUNDERPERFORM + S_WRATING_NUMOFSELL)
            self.basic_data = basic_data.loc[:,basic_data.columns.intersection(self.stock_id)]
            
        # 预测机构家数同比增长率   NUM_EST_INST
        if factor_name == 'NUM_EST_INST':
            
            basic_data = DataUtils.get_daily_info('stock', 'NUM_EST_INST_{}'.format(FY))
            self.basic_data = basic_data.loc[:,basic_data.columns.intersection(self.stock_id)]
            
        # 主营业务收入调高家数占比同比   MAIN_BUS_INC_UPGRADE
        if factor_name == 'MAIN_BUS_INC_UPGRADE':
            
            MAIN_BUS_INC_UPGRADE = DataUtils.get_daily_info('stock', 'MAIN_BUS_INC_UPGRADE_{}'.format(FY))
            MAIN_BUS_INC_DOWNGRADE = DataUtils.get_daily_info('stock', 'MAIN_BUS_INC_DOWNGRADE_{}'.format(FY))
            MAIN_BUS_INC_MAINTAIN = DataUtils.get_daily_info('stock', 'MAIN_BUS_INC_MAINTAIN_{}'.format(FY))
            basic_data = MAIN_BUS_INC_UPGRADE / (MAIN_BUS_INC_UPGRADE + MAIN_BUS_INC_DOWNGRADE + MAIN_BUS_INC_MAINTAIN)
            self.basic_data = basic_data.loc[:,basic_data.columns.intersection(self.stock_id)]
            
        # 净利润调高家数占比同比    NET_PROFIT_UPGRADE
        if factor_name == 'NET_PROFIT_UPGRADE':
            
            NET_PROFIT_UPGRADE = DataUtils.get_daily_info('stock', 'NET_PROFIT_UPGRADE_{}'.format(FY))
            NET_PROFIT_DOWNGRADE = DataUtils.get_daily_info('stock', 'NET_PROFIT_DOWNGRADE_{}'.format(FY))
            NET_PROFIT_MAINTAIN = DataUtils.get_daily_info('stock', 'NET_PROFIT_MAINTAIN_{}'.format(FY))
            basic_data = NET_PROFIT_UPGRADE / (NET_PROFIT_UPGRADE + NET_PROFIT_DOWNGRADE + NET_PROFIT_MAINTAIN)
            self.basic_data = basic_data.loc[:,basic_data.columns.intersection(self.stock_id)]
                        
        # 读取个股相应的市值数据
        self.S_DQ_MV = DataUtils.get_daily_info('stock', 'S_DQ_MV')
        self.S_DQ_MV = self.S_DQ_MV.loc[:, self.S_DQ_MV.columns.intersection(self.stock_id)]

        # 生成月频时间索引
        if self.basic_data.index[-1].month == 12:
            dates = self.gen_panel_dates('2007-01-01', '{}-01-01'.format(
                self.basic_data.index[-1].year + 1))

        else:
            dates = self.gen_panel_dates('2007-01-01', '{}-{}-01'.format(
                self.basic_data.index[-1].year,
                self.basic_data.index[-1].month + 1))

        self.monthly_dates = dates.tolist()
        
        # 生成日频时间索引
        daily_dates=pd.date_range('2007-01-01',self.basic_data.index[-1],freq='D')
        self.S_DQ_MV = self.S_DQ_MV.reindex(daily_dates,method='ffill')
        self.basic_data = self.basic_data.reindex(daily_dates,method='ffill')
    
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
    # 获取特定截面满足筛选条件的数据
    # [输入]
    # panel_date     目标截面日期
    # target_stock   目标个股集合
    # -------------------------------------------------------------------------
    def get_valid_panel_data(self, panel_date, target_stock):
        
        # 拼装截面信息
        df_panel = pd.DataFrame(
            {'est_data' : self.basic_data.loc[panel_date, target_stock], 
             'size': self.S_DQ_MV.loc[panel_date, target_stock],
             'indus_belong' : self.indus_belong.loc[panel_date, target_stock]},
             index=target_stock)
  
        # 去除空值
        df_panel = df_panel.dropna()
   
        return df_panel
        

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
    # 行业指标计算程序
    # [输入]
    # panel_dates    调仓日期序列
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
            
            # 获取截面股票数据
            df_cur = model.get_valid_panel_data(date, valid_stocks)
            df_prev = model.get_valid_panel_data(prev_date, valid_stocks)
            
            # 对齐操作二：前后期去除空值后，可能仍然有不完全匹配的地方
            choosed_stock = list(set(df_cur.index) & set(df_prev.index))
            
            # 保留完全对齐的成分股
            df_cur = df_cur.loc[choosed_stock, :]
            df_prev = df_prev.loc[choosed_stock, :]
            
            # 计算当前截面汇总值
            df_diff_data.loc[date, :] = model.merge_panel_ratio(df_cur, df_prev)

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

        # 比例大于0.5为正
        factor[df_diff_data > 0] = 1
        factor[df_diff_data < 0] = -1
        
        factor = factor.astype('float') 
        
        return factor

if __name__ == '__main__':
    
    # 设置一致预期评级指标    
    factor_all = {'一致预期综合评级同比':'WRATING_AVG',
                  '评级调高家数占比同比':'WRATING_UPGRADE',
                  '买入评级家数占比同比':'WRATING_NUMOFBUY',
                  '增持以上评级家数占比同比':'WRATING_NUMOFOUTPERFORM',
                  '预测机构家数同比增长率':'NUM_EST_INST',
                  '主营业务收入调高家数占比同比':'MAIN_BUS_INC_UPGRADE',
                  '净利润调高家数占比同比':'NET_PROFIT_UPGRADE'}
    
    for factor_name, factor_code in factor_all.items():
        
        print('计算指标：', factor_name)
        
        # 初始化
        model = model_rating(factor_code, 'FY2')
        
        # 时间区间设定
        panel_dates = model.gen_panel_dates('2009-01-01', '2021-05-31', 'W')

        # 计算行业指标
        df_factor = model.gen_factor_ratio(panel_dates)        

        # 行业指标存储
        df_factor.to_pickle('gen_factor/result/est_ratings_{}'.format(factor_code))
        