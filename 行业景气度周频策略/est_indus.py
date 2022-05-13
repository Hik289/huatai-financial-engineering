import sys
sys.path.append("..")
import utils.DataUtils as DataUtils
from utils.BaseClass import BaseClass
import pandas as pd
import datetime

# -----------------------------------------------------------------------------
# 一致预期行业指标
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
            self.data = DataUtils.get_daily_info('indus', data_name)
        else:
            
            # 读取FY1数据
            self.data = DataUtils.get_daily_info('indus', data_name[:-3])
            
            # 读取FY2数据
            self.data_FY2 = DataUtils.get_daily_info('indus', data_name[:-6] + 'FY2')
            data_FY2_index = self.data_FY2.index.tolist()

            # 将FY1数据2月和3月替换成FY2
            for i, ii in self.data.iterrows():
                # 有时候FY1、FY2的日期对不齐，加了个判断条件
                if i.month in [2, 3] and i in data_FY2_index:
                    self.data.loc[i] = self.data_FY2.loc[i]
        
    # -------------------------------------------------------------------------
    # 获取每周周日日期
    # [输入]
    # start_date     开始时间
    # end_date       终止时间
    # frequency      周频
    # -------------------------------------------------------------------------
    def gen_panel_dates(self, start_date, end_date, frequency='W'):
        
        # 指标时间区间
        month_end_dates = pd.date_range(start=start_date, end=end_date, freq='W')
        
        return month_end_dates

    # -------------------------------------------------------------------------
    # 行业指标计算，第一种方法
    # [输入]
    # daily_data       所需处理的日频行业数据
    # panel_dates    调仓日期序列
    # -------------------------------------------------------------------------
    def gen_factor(self, daily_data, panel_dates):
        
        # 生成日频时间索引
        daily_date=pd.date_range(panel_dates[0],panel_dates[-1],freq='D')
        daily_data=daily_data.reindex(daily_date,method='ffill')
        
        # 获取周频数据
        weekly_data = daily_data.reindex(index=panel_dates)
        
        # 获取每月末数据
        monthly_end_data=daily_data.resample('M').last()
        
        # 获取去年同期上月末数据
        monthly_end_data=monthly_end_data.shift(12)
        monthly_end_data=monthly_end_data.reindex(panel_dates,method='ffill')

        # 计算相对去年同期上月末的增长
        weekly_data_change=weekly_data-monthly_end_data
        
        # 截取数据
        factor = weekly_data_change.loc[panel_dates, :]  # 时间截取

        # 数据类型转换
        factor = factor.astype('float')

        # 计算因子值
        df_factor = factor.copy()

        df_factor[df_factor.isnull()] = 0
        df_factor[factor > 0] = 1
        df_factor[factor == 0] = 0
        df_factor[factor < 0] = -1
        
        return  df_factor
    
    
if __name__ == '__main__':

    types = ['FY2','CAGR']
    
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
    
                # 每周周日的日期序列
                panel_dates = model.gen_panel_dates('2009-01-01', '2021-05-31', 'W')
    
                df_factor = model.gen_factor(model.data, panel_dates)
        
                df_factor.to_pickle('gen_factor/result/est_indus_{}'.format(data_name))
                
                print('指标计算成功：', data_type, data_name)

            except:
                print('指标计算失败：', data_type, data_name)