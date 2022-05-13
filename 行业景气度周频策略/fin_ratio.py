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
    # transform   利润表数据加工方式：ttm（滚动平均）、qfa（单季度）、orig（原始累计）
    # -------------------------------------------------------------------------
    def __init__(self, factor, target):
        
        # 初始化父对象
        super().__init__()
        
        # 指标名称记录
        self.factor = factor
       
        
        # 分子
        if factor == 'quick':
            numerator = DataUtils.get_quarterly_info('stock',target[0][0])-\
                DataUtils.get_quarterly_info('stock',target[0][1])
                
        elif factor == 'cashtocurrentdebt':
            numerator = DataUtils.get_quarterly_info('stock',target[0][0])+\
                DataUtils.get_quarterly_info('stock',target[0][1])+DataUtils.get_quarterly_info('stock',target[0][2])
                
        else:
            numerator = DataUtils.get_quarterly_info('stock', target[0])
        
        
        # 根据参数设置，转换成TTM值或单季度值
        if target[1] == 'ttm':
            self.numerator = self.transform_ttm_in_wind_way(numerator)
        elif target[1] == 'qfa':
            self.numerator = self.transform_qfa_in_wind_way(numerator)
        elif target[1] == 'orig':
            self.numerator = numerator  
        
        
        # 分母
        if factor=='nptocostexpense':
            
            denominator1 = DataUtils.get_quarterly_info('stock',target[2][0])
            denominator2 = DataUtils.get_quarterly_info('stock',target[2][1])
            denominator3 = DataUtils.get_quarterly_info('stock',target[2][2])
            denominator4 = DataUtils.get_quarterly_info('stock',target[2][3])
            denominator = denominator1 + denominator2 + denominator3 + denominator4
            
        elif factor in ['capitalizedtoda','ocftocf','ocftoquickdebt']:
            denominator1 = DataUtils.get_quarterly_info('stock',target[2][0])
            denominator2 = DataUtils.get_quarterly_info('stock',target[2][1])
            denominator3 = DataUtils.get_quarterly_info('stock',target[2][2])  
            denominator = denominator1 + denominator2 + denominator3
        
        elif factor in ['operatecaptialturn','ocftointerest','ebittointerest']:
            denominator = DataUtils.get_quarterly_info('stock',target[2][0])-DataUtils.get_quarterly_info('stock',target[2][1])
        
        else:
            denominator = DataUtils.get_quarterly_info('stock', target[2])
       
        
        # 根据参数设置，转换成TTM值或单季度值
        if target[3] == 'ttm':
            self.denominator = self.transform_ttm_in_wind_way(denominator)
        elif target[3] == 'qfa':
            self.denominator = self.transform_qfa_in_wind_way(denominator)
        elif target[3] == 'mean':
            self.denominator = denominator.rolling(4).mean()
        elif target[3] == 'orig':
            self.denominator = denominator  
           
        # 分子数据
        self.numerator = numerator
        
        # 分母数据
        self.denominator = denominator
        
        if factor == 'grossprofitmargin':
            self.numerator = self.denominator - self.numerator 
            
        self.numerator=pd.DataFrame(self.numerator.stack(),columns=['value'])
        self.denominator=pd.DataFrame(self.denominator.stack(),columns=['value'])
        
        # 整理股票的行业属性
        stock_ind=DataUtils.get_daily_info('stock', 'indus_belong')
        stock_ind=pd.DataFrame(stock_ind.stack(),columns=['ind'])
        stock_ind.index.names=['date','stock_code']
        stock_ind=stock_ind[~stock_ind.isnull()]
        stock_ind=stock_ind[stock_ind['ind']!='nan']
        self.stock_ind=stock_ind
        
        # 获取财报的发布日期
        ANNOUN_DATE=DataUtils.get_quarterly_info('stock', 'ANNOUN_DATE')
        ANNOUN_DATE=pd.DataFrame(ANNOUN_DATE.stack(),columns=['ann_dt'])
        ANNOUN_DATE['ann_dt']=ANNOUN_DATE['ann_dt'].apply(lambda x: str(x)[0:4]+'-'+str(x)[4:6]+'-'+str(x)[6:])
        ANNOUN_DATE['ann_dt']=ANNOUN_DATE['ann_dt'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
        self.ANNOUN_DATE=ANNOUN_DATE
       
        # 设定财报发布的截止日期
        self.report_date_dic={'06-30':['08-31'],'09-30':['10-31'],'03-31':['04-30']}
        
    # -------------------------------------------------------------------------
    # 按照财报发布日期整理财报数据
    # [输入]
    # prop_data     按照财报期末日整理的财报数据
    # -------------------------------------------------------------------------
    def get_prop_data(self, prop_data):
        
        prop_data=pd.merge(prop_data,self.ANNOUN_DATE,left_index=True,right_index=True,how='left')
        prop_data.set_index('ann_dt',append=True,inplace=True)
        prop_data.index.names=['report_period','stock_code','date']
        prop_data=prop_data.loc[prop_data.index.dropna()]
        
        return prop_data
    
    # -------------------------------------------------------------------------
    # 获取所有信号日期，在每日获取在当日所能获得的所有有关该财报季的财务数据
    # [输入]
    # panel_dates     所有季度末日期（四季度除外）
    # before          获取财报发布截止日前几周的数据
    # after           获取财报发布截止日后几周的数据
    # -------------------------------------------------------------------------
    def get_valid_panel_data(self, panel_dates, before, after):
        
        self.numerator=self.get_prop_data(self.numerator)
        self.denominator=self.get_prop_data(self.denominator)
        prop_data=pd.merge(self.numerator,self.denominator,left_index=True,right_index=True,how='outer')
        prop_data=prop_data.swaplevel('date','stock_code')
        prop_data.sort_index(inplace=True)
        prop_data.columns=['nominator','denominator']
        
        # 将分子分母按照财报发布时间整理
        prop_data.reset_index('report_period',inplace=True)
        
        # 所有生成信号的日期
        total_signal_date=[]
        
        # 财务数据按照财报的发布日期整理  
        total_signal_df=pd.DataFrame()
        
        # 财务数据按照财报期末日整理
        total_financial_df=pd.DataFrame()
        
        for s in range(len(panel_dates)):
            j=panel_dates[s]

            # 获取相关季度的财报
            financial_data=prop_data[prop_data['report_period']==j.strftime('%Y-%m-%d')]
            financial_data.drop(columns='report_period',inplace=True)
            
            # 获取财报截止日期
            report_ddl=str(j.year)+'-'+self.report_date_dic[j.strftime('%Y-%m-%d')[5:]][0]
            report_ddl=datetime.strptime(report_ddl, '%Y-%m-%d')
            
            # 生成周频日期，从中选取每周日生成信号
            weekly_date=pd.date_range(financial_data.index[0][0],report_ddl+timedelta(days=80),freq='W')
            
            # 获取该财报期信号日日期
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
                # 到该信号日为止的所有该财报期财报数据
                present_data=financial_data[financial_data.index[0][0]:signal_date[i]]
                present_data.reset_index('date',drop=True,inplace=True)
                present_data['date']=signal_date[i]
                present_data=present_data.set_index('date',append=True).swaplevel('date','stock_code')
                signal_value.loc[signal_date[i],['nominator','denominator']]=present_data[['nominator','denominator']]
             
            signal_value['report_period']=j
            
            financial_data.reset_index('date',drop=True,inplace=True)
            financial_data['date']=j
            financial_data=financial_data.set_index('date',append=True).swaplevel('date','stock_code')
            
            # 按财报发布日期整理
            total_signal_df=pd.concat([total_signal_df,signal_value],axis=0)
            
            # 按财报期末日整理
            total_financial_df=pd.concat([total_financial_df,financial_data],axis=0)

        # 去除前三个季度，由于没有同比数据
        total_signal_date=total_signal_date[18:]

        return total_signal_date,total_signal_df,total_financial_df

    # -------------------------------------------------------------------------
    # 将个股季频数据转换为行业数据，计算增量
    # [输入]
    # panel_dates   所有季度末日期（四季度除外）
    # delta_method  增量计算方式，yoy(同比), qoq(环比)
    # -------------------------------------------------------------------------
    def quaterly_expo_merged_by_stock(self, panel_dates, delta_method, before, after):

        total_signal_date,total_signal_df,total_financial_df = self.get_valid_panel_data(panel_dates,before,after)
        
        # 计算同比增长率
        total_growth_df=pd.DataFrame(index=total_signal_date,columns=set(self.stock_ind['ind'].values),data=0)
        total_growth_df=pd.DataFrame(total_growth_df.stack(),columns=['growth'])
        total_growth_df.index.names=['date','ind']
        
        # 环比对应的财报期末日
        prev_date_dic={'06-30':[int(0),'03-31'],'09-30':[int(0),'06-30'],'03-31':[int(-1),'09-30']}
        
        # 计算同比或者环比增长率
        for i in total_signal_date:

            df_cur=total_signal_df.loc[i]
            
            if delta_method=='yoy':
                prev_date=str(df_cur.iloc[0]['report_period'].year-1)+str(df_cur.iloc[0]['report_period'])[4:10]
            else:
                prev_date=str(df_cur.iloc[0]['report_period'].year+prev_date_dic[str(df_cur.iloc[0]['report_period'])[5:10]][0])+'-'+prev_date_dic[str(df_cur.iloc[0]['report_period'])[5:10]][1]
            
            prev_date=datetime.strptime(prev_date,'%Y-%m-%d')
            
            #保留行业属性不变的股票
            df_ind_belong=self.stock_ind.loc[i]
            df_ind_belong['prev_ind']=self.stock_ind.loc[prev_date]['ind']
            valid_stock=df_ind_belong[df_ind_belong['ind']==df_ind_belong['prev_ind']].index
            df_cur=df_cur.loc[valid_stock]
            
            # 获取上一期数据
            df_prev=total_financial_df.loc[prev_date]
            df_cur[['nominator_lag1','denominator_lag1']]=df_prev[['nominator','denominator']]
            df_cur.dropna(axis=0,how='any',inplace=True)

            # 防止数据过大，将数据以亿作为单位，合成行业指标
            df_cur[['nominator_lag1','denominator_lag1','nominator','denominator']]/=1.0e8
            df_cur.set_index('ind',append=True,inplace=True)
            df_ind_cur=df_cur.groupby('ind').sum()
            
            # 计算每期的行业景气度，行业景气度同比/环比增长率
            df_ind_cur['index']=df_ind_cur['nominator']/df_ind_cur['denominator']
            df_ind_cur['index_lag1']=df_ind_cur['nominator_lag1']/df_ind_cur['denominator_lag1']
            df_ind_cur['growth']=df_ind_cur['index']-df_ind_cur['index_lag1']
            df_ind_cur['date']=i
            df_ind_cur=df_ind_cur.set_index('date',append=True).swaplevel('date','ind')
            total_growth_df.loc[i,'growth']=df_ind_cur['growth']

        return total_growth_df

    # -------------------------------------------------------------------------
    # 行业指标计算过程
    # [输入]
    # start_date  开始时间
    # end_date    结束时间
    # delta_method  增量计算方式，yoy(同比), qoq(环比)
    # factor_logic   等于1时因子逻辑为正，等于0时因子逻辑为负
    # -------------------------------------------------------------------------
    def gen_factor(self, start_date, end_date, delta_method, factor_logic, before, after):
      
        # 生成所有财报期末日（除四季度外）
        panel_dates=[]
        for i in pd.date_range(start_date,end_date,freq='Q'):
            if i.strftime('%Y-%m-%d')[6]!='2':
                panel_dates.append(i)
          
        total_growth_df = self.quaterly_expo_merged_by_stock(panel_dates, delta_method, before, after)
        total_growth_df=total_growth_df.unstack()
        total_growth_df.columns=total_growth_df.columns.droplevel()
        
        df_factor = pd.DataFrame(np.zeros_like(total_growth_df.values),
                                  index=total_growth_df.index, columns=total_growth_df.columns)
        
        if self.factor in ['FIX_ASSETS', 'NET_CASH_FLOWS_OPER_ACT', 'NET_CASH_FLOWS_INV_ACT', 
                           'NET_CASH_FLOWS_FNC_ACT', 'NET_INCR_CASH_CASH_EQU', 'FREE_CASH_FLOW']:
            total_growth_df.loc[:, ['银行', '综合金融', '证券Ⅱ', '保险Ⅱ', '多元金融']] = np.nan
            
        # 差值大于零的指标做多，反之做空
        df_factor[total_growth_df > 0] = 1*factor_logic
        df_factor[total_growth_df < 0] = -1*factor_logic
        
        # 将财务因子的信号日期拓展至所有信号日期，原因子数据缺失按0填充
        total_signal_dates=pd.date_range(start_date, end_date, freq='W')
        total_indus=DataUtils.get_daily_info('indus', 'close').columns
        new_factor=pd.DataFrame(index=total_signal_dates,columns=total_indus,data=0)        
        new_factor.loc[df_factor.index,:]=df_factor.loc[df_factor.index,:]
        df_factor=new_factor
        
        return df_factor
    

    
if __name__ == '__main__':
                                
    # 生成指标的起止时间
    start_date = '2007-03-31'
    end_date = '2021-05-31'

    # 获取财报发布截止日前几周、后几周的数据
    before=2
    after=4
    
# =============================================================================
#  盈利能力指标       
# =============================================================================
# 销售净利率     netprofitmargin
# 销售毛利率     grossprofitmargin
# 净资产收益率    roe
# 总资产净利率    roa
# 成本费用利润率  nptocostexpense    
# 销售费用率     operateexpensetogr
# 财务费用率     finaexpensetogr
# 管理费用率     adminexpensetogr

    factor_dict = {'netprofitmargin':['NET_PROFIT_INCL_MIN_INT_INC','ttm','OPER_REV','ttm', 'qoq',1],
                    'grossprofitmargin':['LESS_OPER_COST','ttm','OPER_REV','ttm', 'qoq',1],
                    'roe':['NET_PROFIT_EXCL_MIN_INT_INC','ttm','TOT_SHRHLDR_EQY_EXCL_MIN_INT','orig', 'qoq',1],
                    'roa':['NET_PROFIT_EXCL_MIN_INT_INC','ttm','TOT_ASSETS','orig', 'qoq',1],
                    'nptocostexpense':['NET_PROFIT_INCL_MIN_INT_INC','ttm',['LESS_OPER_COST','LESS_SELLING_DIST_EXP',
                                      'LESS_GERL_ADMIN_EXP','LESS_FIN_EXP'],'ttm','qoq',1],
                    'operateexpensetogr':['LESS_SELLING_DIST_EXP','ttm','TOT_OPER_REV','ttm', 'qoq',-1],
                    'finaexpensetogr':['LESS_FIN_EXP','ttm','TOT_OPER_REV','ttm', 'qoq',-1],
                    'adminexpensetogr':['LESS_GERL_ADMIN_EXP','ttm','TOT_OPER_REV','ttm', 'qoq',-1]}
    
    
    for factor, data_list in factor_dict.items():
        
        print(factor)
        
        # 模型初始化
        model = fin_model(factor, data_list)
    
        # 行业指标计算
        df_factor = model.gen_factor(start_date, end_date, data_list[4], data_list[5], before, after)
                
        # 写入文件
        df_factor.to_pickle('gen_factor/result/{}'.format(factor))

# =============================================================================
#  收益质量指标       
# =============================================================================                  
# 经营活动净收益/利润总额  operateincometoebt
# 所得税/利润总额  taxtoebt

    factor_dict = {'operateincometoebt':['S_FA_OPERATEINCOME','ttm','TOT_PROFIT','ttm', 'qoq',1],
                    'taxtoebt':['INC_TAX','ttm','TOT_PROFIT','ttm', 'qoq',-1]}
    
    for factor, data_list in factor_dict.items():
        
        print(factor)
        
        # 模型初始化
        model = fin_model(factor, data_list)
        
        # 行业指标计算
        df_factor = model.gen_factor(start_date, end_date, data_list[4], data_list[5], before, after)
                
        # 写入文件
        df_factor.to_pickle('gen_factor/result/{}'.format(factor))
        
# =============================================================================
#  现金流量指标       
# =============================================================================
# 销售收现比     salescashintoor
# 销售现金比率     ocftoor
# 经营活动产生的现金流量净额/经营活动净收益    ocftooperateincome
# 净利润现金含量    netprofitcashcover
# 资本支出/折旧和摊销  capitalizedtoda    
# 经营活动产生的现金流量净额占比     ocftocf
# 全部资产现金回收率     ocftoasset
# 现金股利保障倍数  ocftodividend
        
    factor_dict = {'salescashintoor':['CASH_RECP_SG_AND_RS','ttm','OPER_REV','ttm', 'qoq',1],
                    'ocftoor':['NET_CASH_FLOWS_OPER_ACT','ttm','OPER_REV','ttm', 'qoq',-1],
                    'ocftooperateincome':['NET_CASH_FLOWS_OPER_ACT','ttm','S_FA_OPERATEINCOME','ttm', 'qoq',-1],
                    'netprofitcashcover':['NET_CASH_FLOWS_OPER_ACT','ttm','NET_PROFIT_EXCL_MIN_INT_INC','ttm', 'qoq',-1],
                    'capitalizedtoda':['CASH_PAY_ACQ_CONST_FIOLTA','ttm',['DEPR_FA_COGA_DPBA','AMORT_INTANG_ASSETS',
                                        'AMORT_LT_DEFERRED_EXP'],'ttm','qoq',1],
                    'ocftocf':['NET_CASH_FLOWS_OPER_ACT','ttm',['NET_CASH_FLOWS_OPER_ACT','NET_CASH_FLOWS_INV_ACT','NET_CASH_FLOWS_FNC_ACT'],
                                'ttm', 'qoq',-1],
                    'ocftoassets':['NET_CASH_FLOWS_OPER_ACT','ttm','TOT_ASSETS','orig', 'qoq',-1],
                    'ocftodividend':['NET_CASH_FLOWS_OPER_ACT','ttm','COMSHARE_DVD_PAYABLE','ttm','qoq',-1]}
    
    for factor, data_list in factor_dict.items():
        
        print(factor)
        
        # 模型初始化
        model = fin_model(factor, data_list)
        
        # 行业指标计算
        df_factor = model.gen_factor(start_date, end_date, data_list[4], data_list[5], before, after)
                
        # 写入文件
        df_factor.to_pickle('gen_factor/result/{}'.format(factor))
        

# =============================================================================
#  资本结构指标       
# =============================================================================
# 资产负债率     debttoassets

    factor_dict = {'debttoassets':['TOT_LIAB','orig','TOT_ASSETS','orig', 'yoy',1]}
    for factor, data_list in factor_dict.items():
        
        print(factor)
        
        # 模型初始化
        model = fin_model(factor, data_list)
        
        # 行业指标计算
        df_factor = model.gen_factor(start_date, end_date, data_list[4], data_list[5], before, after)
                
        # 写入文件
        df_factor.to_pickle('gen_factor/result/{}'.format(factor))

# =============================================================================
#  偿债能力指标       
# =============================================================================
# 流动比率  current
# 速动比率  quick
# 现金比率  cashtocurrentdebt 
# 现金到期债务比   ocftoquickdebt
# 现金流动负债比   ocftoshortdebt
# 现金流量利息保障倍数    ocftointerest
# 净资产负债率     debttoequity
# 息税折旧摊销前利润/负债  ebitdatodebt
# 现金债务总额比   ocftodebt
# 已获利息倍数    ebittointerest

    factor_dict = {'current':['TOT_CUR_ASSETS','orig','TOT_CUR_LIAB','orig', 'yoy',-1],
                    'quick':[['TOT_CUR_ASSETS','INVENTORIES'],'orig','TOT_CUR_LIAB','orig','yoy',-1],
                    'cashtocurrentdebt':[['MONETARY_CAP','TRADABLE_FIN_ASSETS','NOTES_RCV'],'orig',
                                        'TOT_CUR_LIAB','orig','yoy',-1],
                    'ocftoquickdebt':['NET_CASH_FLOWS_OPER_ACT','ttm',
                                      ['ST_BORROW','NON_CUR_LIAB_DUE_WITHIN_1Y','NOTES_PAYABLE'],'orig','qoq',-1],
                    'ocftoshortdebt':['NET_CASH_FLOWS_OPER_ACT','ttm','TOT_CUR_LIAB','ttm', 'qoq',-1],
                    'ocftointerest':['NET_CASH_FLOWS_OPER_ACT','ttm',['EBIT','TOT_PROFIT'],'ttm', 'qoq',1],
                    'debttoequity':['TOT_LIAB','orig','TOT_SHRHLDR_EQY_EXCL_MIN_INT','orig', 'yoy',1],
                    'ebitdatodebt':['EBITDA','ttm','TOT_LIAB','orig', 'qoq',-1],
                    'ocftodebt':['NET_CASH_FLOWS_OPER_ACT','ttm','TOT_CUR_LIAB','orig','qoq',-1],
                    'ebittointerest':['EBIT','ttm',['EBIT','TOT_PROFIT'],'ttm','qoq',1]}
    
        
    for factor, data_list in factor_dict.items():
        
        print(factor)
        
        # 模型初始化
        model = fin_model(factor, data_list)
        
        # 行业指标计算
        df_factor = model.gen_factor(start_date, end_date, data_list[4], data_list[5], before, after)
                
        # 写入文件
        df_factor.to_pickle('gen_factor/result/{}'.format(factor))
        
# =============================================================================
#  营运能力指标       
# =============================================================================
# 存货周转率  invturn
# 总资产周转率  assetsturn
# 应收账款周转率  arturn 
# 流动资产周转率   caturn
# 营运资本周转率   operatecaptialturn
# 固定资产周转率    faturn
# 应付账款周转率     apturn
# 现金周转率  cashturn

    factor_dict = {'invturn':['LESS_OPER_COST','ttm','INVENTORIES','orig', 'qoq',1],
                    'assetsturn':['TOT_OPER_REV','ttm','TOT_ASSETS','orig', 'qoq',1],
                    'arturn':['OPER_REV','ttm','ACCT_RCV','orig', 'qoq',1],
                    'caturn':['TOT_OPER_REV','ttm','TOT_CUR_ASSETS','orig', 'qoq',1],
                    'operatecaptialturn':['TOT_OPER_REV','ttm',['TOT_CUR_ASSETS','TOT_CUR_LIAB'],'orig', 'qoq',1],
                    'faturn':['TOT_OPER_REV','ttm','FIX_ASSETS','orig', 'qoq',1],
                    'apturn':['LESS_OPER_COST','ttm','ACCT_PAYABLE','orig', 'qoq',1],
                    'cashturn':['TOT_OPER_REV','ttm','CASH_CASH_EQU_END_PERIOD','orig','qoq',1]}
  

    for factor, data_list in factor_dict.items():
        
        print(factor)
        
        # 模型初始化
        model = fin_model(factor, data_list)
        
        # 行业指标计算
        df_factor = model.gen_factor(start_date, end_date, data_list[4], data_list[5], before, after)
                
        # 写入文件
        df_factor.to_pickle('gen_factor/result/{}'.format(factor))

