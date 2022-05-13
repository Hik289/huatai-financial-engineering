import sys
sys.path.append("..")
import utils.DataUtils as DataUtils
from utils.BaseClass import BaseClass
import pandas as pd
import numpy as np
import datetime
from datetime import datetime,timedelta
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 定期业绩指标
# -----------------------------------------------------------------------------
class fin_model(BaseClass):
    
    # -------------------------------------------------------------------------
    # 实例化对象，主要用于加载全局数据，避免每次重复加载
    # target_prop 要加载的属性名称，可以是营业收入、利润总额等字段
    # transform   利润表数据加工方式：ttm（滚动平均）、qfa（单季度）、orig（原始累计）
    # -------------------------------------------------------------------------
    def __init__(self, target_prop, transform):
        
        # 初始化父对象
        super().__init__()
        
        # 指标名称记录
        self.factor = target_prop
        
        # 获取对应利润表字段的数据，对应累计值
        prop_data = DataUtils.get_quarterly_info('stock', target_prop)
        
        # 根据参数设置，转换成TTM值或单季度值
        if transform == 'ttm':
            prop_data = self.transform_ttm_in_wind_way(prop_data)
        elif transform == 'qfa':
            prop_data = self.transform_qfa_in_wind_way(prop_data)
        else:
            prop_data = prop_data

        prop_data=pd.DataFrame(prop_data.stack(),columns=['value'])    
            
        # 整理股票的行业属性
        stock_ind=DataUtils.get_daily_info('stock', 'indus_belong')
        stock_ind=pd.DataFrame(stock_ind.stack(),columns=['ind'])
        stock_ind.index.names=['date','stock_code']
        stock_ind=stock_ind[~stock_ind.isnull()]
        stock_ind=stock_ind[stock_ind['ind']!='nan']
        self.stock_ind=stock_ind
        
        # 获取财报的发布日期
        ANNOUN_DATE=DataUtils.get_quarterly_info('stock', 'ANN_DT')
        ANNOUN_DATE=pd.DataFrame(ANNOUN_DATE.stack(),columns=['ann_dt'])
        ANNOUN_DATE['ann_dt']=ANNOUN_DATE['ann_dt'].apply(lambda x: str(x)[0:4]+'-'+str(x)[4:6]+'-'+str(x)[6:])
        ANNOUN_DATE['ann_dt']=ANNOUN_DATE['ann_dt'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
        self.ANNOUN_DATE=ANNOUN_DATE
        
        # 按照财报发布时间整理财报数据
        prop_data=pd.merge(prop_data,ANNOUN_DATE,left_index=True,right_index=True,how='left')
        prop_data.set_index('ann_dt',append=True,inplace=True)
        prop_data.index.names=['report_period','stock_code','date']
        prop_data=prop_data.reset_index('report_period').swaplevel('date','stock_code')
        prop_data.sort_index(inplace=True)
        prop_data=prop_data[~prop_data.index.duplicated(keep='last')]
        prop_data=prop_data.loc[prop_data.index.dropna()]
        self.prop_data=prop_data
        
        # 设定财报发布截止日期
        self.report_date_dic={'06-30':['08-31'],'09-30':['10-31'],'03-31':['04-30']}
    
    # -------------------------------------------------------------------------
    # 获取所有信号日期，在每日获取在当日所能获得的所有有关该财报季的财务数据
    # [输入]
    # panel_dates     所有季度末日期（四季度除外）
    # before  获取财报发布截止日前几周的数据
    # after   获取财报发布截止日后几周的数据
    # -------------------------------------------------------------------------
    def get_valid_panel_data(self, panel_dates, before, after):
        
        # 所有信号日期
        total_signal_date=[]
        # 按财报发布日期整理  
        total_signal_df=pd.DataFrame()
        # 按季度末日期整理
        total_financial_df=pd.DataFrame()
        
        for s in range(len(panel_dates)):
            j=panel_dates[s]
    
            # 获取相关季度的财报
            financial_data=self.prop_data[self.prop_data['report_period']==j.strftime('%Y-%m-%d')]
            financial_data.drop(columns='report_period',inplace=True)
            
            # 获取财报截止日期
            report_ddl=str(j.year)+'-'+self.report_date_dic[j.strftime('%Y-%m-%d')[5:]][0]
            report_ddl=datetime.strptime(report_ddl, '%Y-%m-%d')
            
            # 生成周频日期，从中选取每周日生成信号
            weekly_date=pd.date_range(financial_data.index[0][0],report_ddl+timedelta(days=80),freq='W')
            
            # 该财报季信号日期
            signal_date=[]
            
            # 计算该财报季第一份财报发布时间至财报季末共有几个星期日
            time_lag=report_ddl-timedelta(days=report_ddl.weekday()+1)-datetime.strptime('2021-04-16','%Y-%m-%d')
            max_before=time_lag.days//7+1
            
            # before定义为before与max_before的最小值
            before = before if max_before>=before else max_before
            
            # 生成财报发布截止日前信号日期
            for i in range(0,before):
                sig_date=report_ddl-timedelta(days=report_ddl.weekday()+1)-timedelta(days=7*(before-1-i))
                if sig_date in self.stock_ind.index:
                    # 所有信号日期
                    total_signal_date.append(sig_date)
                    # 该财报季信号日期
                    signal_date.append(sig_date)
            
            # 生成财报发布截止日后信号日期
            for i in range(0,after):
                sig_date=report_ddl+timedelta(days=(6-report_ddl.weekday()))+timedelta(days=7*i)
                if sig_date in self.stock_ind.index:
                    # 所有信号日期
                    total_signal_date.append(sig_date)
                    # 该财报季信号日期
                    signal_date.append(sig_date)
            
            # 将财报数据与股票行业分类对应    
            signal_value=self.stock_ind.loc[signal_date]

            for i in range(0,len(signal_date)):
                # 到该信号日为止的所有该季度财报数据
                present_data=financial_data[financial_data.index[0][0]:signal_date[i]]
                present_data.reset_index('date',drop=True,inplace=True)
                present_data['date']=signal_date[i]
                present_data=present_data.set_index('date',append=True).swaplevel('date','stock_code')
                signal_value.loc[signal_date[i],'value']=present_data['value']
             
            signal_value['report_period']=j
            
            financial_data.reset_index('date',drop=True,inplace=True)
            financial_data['date']=j
            financial_data=financial_data.set_index('date',append=True).swaplevel('date','stock_code')
            
            # 按财报发布日期整理
            total_signal_df=pd.concat([total_signal_df,signal_value],axis=0)
            # 按季度末日期整理
            total_financial_df=pd.concat([total_financial_df,financial_data],axis=0)

        # 去除前三个季度，由于没有同比数据
        total_signal_date=total_signal_date[24:]
   
        return total_signal_date,total_signal_df,total_financial_df    

    # -------------------------------------------------------------------------
    # 行业指标计算过程
    # [输入]
    # start_date  开始时间
    # end_date    结束时间
    # before  获取财报发布截止日前几周的数据
    # after   获取财报发布截止日后几周的数据
    # -------------------------------------------------------------------------
    def gen_factor(self,start_date, end_date, before, after):
        
        # 生成所有财报的季度末日期
        panel_dates=[]
        for i in pd.date_range(start_date,end_date,freq='Q'):
            if i.strftime('%Y-%m-%d')[6]!='2':
                panel_dates.append(i)
        
        total_signal_date,total_signal_df,total_financial_df = self.get_valid_panel_data(panel_dates, before, after)
        
        # 计算同比增长率
        total_growth_df=pd.DataFrame(index=total_signal_date,columns=set(self.stock_ind['ind'].values),data=0)
        total_growth_df=pd.DataFrame(total_growth_df.stack(),columns=['growth'])
        total_growth_df.index.names=['date','ind']
        
        for i in total_signal_date:
            df_cur=total_signal_df.loc[i]
            
            # 去年同期财报季
            prev_date=str(df_cur.iloc[0]['report_period'].year-1)+str(df_cur.iloc[0]['report_period'])[4:10]
            prev_date=datetime.strptime(prev_date,'%Y-%m-%d')
            
            # 保留行业属性不变的股票
            df_ind_belong=self.stock_ind.loc[i]
            df_ind_belong['prev_ind']=self.stock_ind.loc[prev_date]['ind']
            valid_stock=df_ind_belong[df_ind_belong['ind']==df_ind_belong['prev_ind']].index
            df_cur=df_cur.loc[valid_stock]
            
            # 获取去年同期和当期数据
            df_prev=total_financial_df.loc[prev_date]
            df_cur['prev_value']=df_prev['value']
            df_cur=df_cur[df_cur['value'].notna()]
            df_cur=df_cur[df_cur['prev_value'].notna()]

            ## 防止数据过大，将数据以亿作为单位，合成行业指标
            df_cur['value']/=1.0e8
            df_cur['prev_value']/=1.0e8
            df_cur.set_index('ind',append=True,inplace=True)
            df_ind_cur=df_cur.groupby('ind').sum()
            
            # 计算行业景气度同比增长率
            df_ind_cur['growth']=df_ind_cur['value']/df_ind_cur['prev_value']-1
            df_ind_cur['date']=i
            df_ind_cur=df_ind_cur.set_index('date',append=True).swaplevel('date','ind')
            total_growth_df.loc[i,'growth']=df_ind_cur['growth']
        
        # 计算同比增长率的环比增量
        total_growth_diff=total_growth_df-total_growth_df.groupby('ind').shift(6)
        total_growth_diff=total_growth_diff.unstack()
        total_growth_diff.columns=total_growth_diff.columns.droplevel()
        
        df_factor = pd.DataFrame(np.zeros_like(total_growth_diff.values),
                                  index=total_growth_diff.index, columns=total_growth_diff.columns)
        
        if self.factor in ['FIX_ASSETS', 'NET_CASH_FLOWS_OPER_ACT', 'NET_CASH_FLOWS_INV_ACT', 
                           'NET_CASH_FLOWS_FNC_ACT', 'NET_INCR_CASH_CASH_EQU', 'FREE_CASH_FLOW']:
            total_growth_diff.loc[:, ['银行', '综合金融', '证券Ⅱ', '保险Ⅱ', '多元金融']] = np.nan
            
        # 差值大于零的指标做多，反之做空
        df_factor[total_growth_diff > 0] = 1
        df_factor[total_growth_diff < 0] = -1
        
        # 将财务因子的信号日期拓展至所有信号日期，原因子数据缺失按0填充
        total_signal_dates=pd.date_range(start_date, end_date, freq='W')
        total_indus=DataUtils.get_daily_info('indus', 'close').columns
        new_factor=pd.DataFrame(index=total_signal_dates,columns=total_indus,data=0)        
        new_factor.loc[df_factor.index,:]=df_factor.loc[df_factor.index,:]
        df_factor=new_factor
        
        return df_factor
    

if __name__ == '__main__':
    
    start_date = '2007-03-31'
    end_date = '2021-05-31'
    
# =============================================================================
# 成长能力指标
# =============================================================================

    # 营业收入  OPER_REV
    # 归母净利润     NET_PROFIT_EXCL_MIN_INT_INC
    # 净利润   NET_PROFIT_INCL_MIN_INT_INC
    # 利润总额  TOT_PROFIT
    # 固定资产    FIX_ASSETS
    # 经营活动产生的现金流净额  NET_CASH_FLOWS_OPER_ACT
    # 投资活动产生的现金流净额  NET_CASH_FLOWS_INV_ACT
    # 筹资活动产生的现金流净额  NET_CASH_FLOWS_FNC_ACT
    # 现金及现金等价物净增加额  NET_INCR_CASH_CASH_EQU
    # 自由现金流     FREE_CASH_FLOW

    # 设置财务指标    
    target_all = {'NET_PROFIT_EXCL_MIN_INT_INC': 'net_profit_excl',
                  'NET_PROFIT_INCL_MIN_INT_INC': 'net_profit_incl',
                  'TOT_PROFIT': 'tot_profit', 
                  'OPER_REV': 'oper_rev',
                  'FIX_ASSETS': 'fix_assets',
                  'NET_CASH_FLOWS_OPER_ACT': 'oper_cash',
                  'NET_CASH_FLOWS_INV_ACT': 'inv_cash',
                  'NET_CASH_FLOWS_FNC_ACT': 'fnc_cash',
                  'NET_INCR_CASH_CASH_EQU': 'incr_cash',
                  'FREE_CASH_FLOW': 'free_cash'}
    
    # 获取财报发布截止日前几周、后几周的数据
    before=2
    after=4
    
    for target_prop, factor_name in target_all.items():
        
        print(target_prop)
        
        transform = 'ttm'    
        
        # 实例化对象，默认采用TTM处理方式
        model = fin_model(target_prop, transform)
    
        # 行业指标计算
        df_factor = model.gen_factor(start_date, end_date, before, after  )
        
        # 将存储因子暴露
        df_factor.to_pickle('gen_factor/result/{}'.format(factor_name))