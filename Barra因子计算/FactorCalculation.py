# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 18:07:41 2021

@author: kipcrk290
"""
import numpy as np
import pandas as pd
import sys
# import pickle
from utils import PreprocessingTools as ppt

class gen_factor:
    
    def __init__(self,arg):
        
        # 指标开始和结束日期
        self.start_date = arg["指标开始日期"]
        self.end_date = arg["指标结束日期"]
        
        # 股票基本信息
        self.stock_info = pd.read_pickle("data/basic/stock_info")
        
        # 中信一级行业归属
        self.indus1_belong = pd.read_pickle("data/daily/stock/indus1_belong")
        
        # 股票收盘价和日收益率
        self.stock_close = pd.read_pickle("data/daily/stock/S_DQ_ADJCLOSE")
        self.daily_ret = self.stock_close.pct_change(fill_method=None)
        
        # 股票换手率
        self.stock_turn = pd.read_pickle("data/daily/stock/S_DQ_TURN")
        
        # 股票总市值
        self.total_mkv = pd.read_pickle("data/daily/stock/S_DQ_MV")
        
        # 股票流通市值
        self.val_mkv = pd.read_pickle("data/daily/stock/S_VAL_MV")
        
        # 指数收盘价
        self.index_close = pd.read_pickle("data/daily/index/S_DQ_CLOSE")
        self.index_close = self.index_close[arg["基准指数代码"]]
        
        # 文件名称映射
        self.name_dict = {"总市值":"total_mkv","流通市值":"val_mkv"}
        
        # 季报索引
        self.quarterLoc = pd.read_pickle('data/quarterly/stock/ANN_DT')
    
    
    # ================================================================================
    # 行业因子
    # ================================================================================
    def IndusFactorExpo(self):
        factor = self.indus1_belong.loc[self.start_date:self.end_date] \
            .reindex(columns=self.stock_info.index)
        return factor
    
    
    # ================================================================================
    # 规模因子
    # category      选择市值类型，["总市值","流通市值"]
    # logflag       是否计算自然对数，bool
    # ================================================================================
    def SizeFactorExpo(self,category,logflag=True):
        
        # 选择市值类型
        data = eval(f"self.{self.name_dict[category]}")
        
        # 调整行列索引
        data = data.loc[self.start_date:self.end_date] \
            .reindex(columns=self.stock_info.index)
        
        # 计算自然对数
        if logflag:
            data[data<1] = np.nan
            factor = np.log(data)
        
        # # 填充空值
        # factor = ppt.fill_missing(data,"fixed")
        
        return factor
    
    
    # ================================================================================
    # 动量因子
    # timewindow    窗口长度，int
    # halflife_days 半衰期天数，int
    # logflag       是否计算自然对数，bool
    # ================================================================================
    def MomentumFactorExpo(self,timewindow,halflife_days=None,logflag=True):
        
        # 计算权重向量
        if halflife_days is not None:
            dampling = 0.5**(np.arange(timewindow,0,-1)/halflife_days)
        else:
            dampling = np.ones(timewindow)
        
        # 调整股票日收益率的行列索引
        data = self.daily_ret.loc[self.start_date:self.end_date] \
            .reindex(columns=self.stock_info.index)
        
        # 计算自然对数
        if logflag:
            data = np.log(1+data)
        
        # 计算加权动量因子
        factor,_ = ppt.rolling_weighted_moment(data,timewindow,weight=dampling)
        
        # # 填充空值
        # factor = ppt.fill_missing(factor,"fixed")
        
        return factor
    
    
    # ================================================================================
    # beta因子（和残差波动率因子）
    # timewindow    窗口长度，int
    # halflife_days 半衰期天数，int
    # ================================================================================
    def BetaFactorExpo(self,timewindow,halflife_days=None):
        
        # 计算权重向量
        if halflife_days is not None:
            dampling = 0.5**(np.arange(timewindow,0,-1)/halflife_days)
        else:
            dampling = np.ones(timewindow)
        
        # 调整股票和基准指数日收益率的行列索引
        indepVar = self.index_close.pct_change(fill_method=None) \
            .loc[self.start_date:self.end_date]
        depVar = self.daily_ret.loc[self.start_date:self.end_date] \
            .reindex(columns=self.stock_info.index)
        
        # 数据检查
        if not indepVar.index.equals(depVar.index):
            print("因子计算失败：检查自变量和因变量的日期是否完全相同")
            sys.exit(0)
        
        # 调整自变量的格式
        indepdf = np.ones([indepVar.shape[0],depVar.shape[1]]) * np.expand_dims(indepVar,1)
        indepdf = pd.DataFrame(indepdf,index=depVar.index,columns=depVar.columns)
        
        # 求解加权线性回归的系数和残差
        alpha,beta,volResid = ppt.rolling_weighted_regression( \
            depVar,indepdf,timewindow,weight=dampling)
        
        # # 填充空值
        # alpha = ppt.fill_missing(alpha,"fixed")
        # beta = ppt.fill_missing(beta,"fixed")
        # volResid = ppt.fill_missing(volResid,"fixed")
        
        return alpha,beta,volResid
    
    
    # ================================================================================
    # 波动率因子
    # name          指标名称，["DASTD","CMRA"]
    # timewindow    窗口长度，int
    # halflife_days 半衰期天数，int
    # logflag       是否计算自然对数，bool
    # num_of_month  滚动计算月数
    # month_ndays   每月的交易日数
    # ================================================================================
    def ResidVolaFactorExpo(self,name,timewindow,num_of_month,halflife_days=None,
                            month_ndays=21,logflag=True):
        
        if name == "DASTD":
        
            # 计算权重向量
            if halflife_days is not None:
                dampling = 0.5**(np.arange(timewindow,0,-1)/halflife_days)
            else:
                dampling = np.ones(timewindow)
            
            # 调整股票日收益率的行列索引
            data = self.daily_ret.loc[self.start_date:self.end_date] \
                .reindex(columns=self.stock_info.index)
            
            # 计算自然对数
            if logflag:
                data = np.log(1+data)
            
            # 计算加权标准差
            _,factor = ppt.rolling_weighted_moment(data,timewindow,weight=dampling)
            
        
        elif name == "CMRA":
            
            # 调整股票收盘价的行列索引
            data = self.stock_close.loc[self.start_date:self.end_date] \
                .reindex(columns=self.stock_info.index)
            
            # 计算总窗口长度
            total_tw = num_of_month*month_ndays+1
            
            # 初始化输出结果
            factor = pd.DataFrame(index=data.index,columns=data.columns)
            
            for i in range(total_tw-1,data.shape[0]):
                
                # 进度条
                ppt.progress_bar(i-total_tw+2,data.shape[0]-total_tw+1)
                
                # 计算累积月收益率
                cur_seq = np.arange(i-total_tw+1,i+1,month_ndays)
                monthly_ret = data.iloc[cur_seq]
                monthly_ret = monthly_ret.iloc[1:] / monthly_ret.iloc[0]
                
                # 计算对数收益率
                monthly_ret = np.log(monthly_ret)
                
                # 负值调整
                monthly_ret[monthly_ret<=-1] = -0.99 #-0.999
                
                # 计算收益率的波动幅度
                factor.iloc[i] = np.log(1+monthly_ret.max()) - np.log(1+monthly_ret.min())
        
        # # 填充空值
        # factor = ppt.fill_missing(factor,"fixed")
        
        return factor
    
    
    # ================================================================================
    # 非线性因子
    # category      选择市值类型，["总市值","流通市值"]
    # issqrt        是否对市值开根
    # ================================================================================
    def NonLinearFactorExpo(self,category,issqrt=False):
        
        # 选择市值类型
        mkv = eval(f"self.{self.name_dict[category]}")
        
        # 调整行列索引
        mkv = mkv.loc[self.start_date:self.end_date] \
            .reindex(columns=self.stock_info.index)
        notnull = mkv.notnull()
        
        # 百分位数去极值
        mkv = ppt.winsorize(mkv,"quantile",axis=1,lower=0,upper=0.95)
        
        # 读取规模因子
        size_factor = self.SizeFactorExpo(category)
        
        # 中位数去极值
        dataX = ppt.winsorize(size_factor,"median-MAD",axis=1,multiplier=3)
        
        # 填充空值
        cate_df = self.indus1_belong.loc[self.start_date:self.end_date] \
            .reindex(columns=self.stock_info.index)
        dataX = ppt.fill_missing(dataX,"category_mean",axis=1,cate_df=cate_df,thres=0)
        dataX = ppt.fill_missing(dataX,"fixed")
        
        # 去掉无效数据
        dataX = dataX[notnull]
        
        # 根据参数设置确定是否对市值开根号
        if issqrt:
            mkv = np.sqrt(mkv)
        
        # 把无效股票的市值置为空值
        mkv = mkv.reindex(columns=dataX.columns)[dataX.notnull()]
        
        # 使用加权均值进行标准化
        dataX_mean = (dataX*mkv).sum(axis=1) / mkv.sum(axis=1).replace(0,np.nan)
        dataX = (dataX-np.repeat(np.expand_dims(dataX_mean,1),len(dataX.columns),1))\
                /np.repeat(np.expand_dims(dataX.std(axis=1).replace(0,np.nan),1),len(dataX.columns),1)
        
        # 将规模因子的三次方对规模因子正交，取残差作为非线性规模因子
        dataY = dataX**3
        factor = pd.DataFrame(index=dataX.index,columns=dataX.columns)
        
        length = factor.shape[0]
        for i,ind in enumerate(factor.index):
            
            # 进度条
            ppt.progress_bar(i+1,length)
            
            # 获取截面数据
            y = dataY.loc[ind]
            x = dataX.loc[ind]
            w = mkv.loc[ind]
            
            # 获取截面上的有效索引
            valid_locs = y.notnull() & x.notnull() & w.notnull()
            _,_,resid = ppt.weighted_regression( \
                y[valid_locs],x[valid_locs],weight=w[valid_locs])
            factor.loc[ind,valid_locs] = resid[:,0]
        
        # # 填充空值
        # factor = ppt.fill_missing(factor,"fixed")
        
        return factor




    
    
    # ================================================================================
    # 市净率因子
    # category      选择市值类型，["总市值","流通市值"]
    # ================================================================================
    def BooktoPriceFactorExpo(self,category,Date):
        
        # 选择市值类型
        data = eval(f"self.{self.name_dict[category]}")
        
        # 调整行列索引
        data = data.loc[self.start_date:self.end_date] \
            .reindex(columns=self.stock_info.index)
        
        #重新赋值，防止引用传递           
        date = Date.copy()
        
        # 读取股票净资产
        tot_equity = pd.read_pickle("data/quarterly/stock/TOT_SHRHLDR_EQY_EXCL_MIN_INT").loc[self.start_date:self.end_date].resample('Y').last()
                
        for i,istock in enumerate(date.columns):
            ppt.progress_bar(i+1,len(date.columns))
            date.loc[:,istock] = date.loc[:,istock].apply(lambda x: np.nan if pd.isnull(x) else tot_equity.loc[x,istock])
        
        # 向后填充
        CE = date.fillna(method='ffill')
        CE = CE.loc[self.start_date:self.end_date]
        
        # 计算账面市值比
        factor = CE/data
        
        # # 填充空值
        # factor = ppt.fill_missing(factor,"fixed")
        
        return factor
    
    
    # ================================================================================
    # 流动性因子
    # category      选择市值类型，["总市值","流通市值"]
    # freq          频率，["M","Q","Y"]
    # month_ndays   每月的交易日数
    # quarter_nmonths    每季度的月数
    # year_nmonths  每年的月数
    # logflag       是否计算自然对数，bool
    # ================================================================================
    def LiquidityFactorExpo(self,freq,month_ndays=21,quarter_nmonths=3,year_nmonths=12,logflag=True):
        
        # 调整行列索引
        data = self.stock_turn.loc[self.start_date:self.end_date] \
            .reindex(columns=self.stock_info.index)
        
        data.dropna(how='all',inplace=True)
        factor = pd.DataFrame(columns=data.columns,index=data.index)
        # 计算区间累计换手率
        if freq == "M":
            nmonths = 1
            for i in range(nmonths* month_ndays-1,len(factor)):
                ppt.progress_bar(i+1-nmonths* month_ndays+1,len(factor)-nmonths* month_ndays+1)
                factor.iloc[i,:] = data.iloc[i-nmonths* month_ndays+1:i,:].mean(axis=0)*nmonths* month_ndays
        elif freq == "Q":
            nmonths = quarter_nmonths
            for i in range(nmonths* month_ndays-1,len(factor)):
                ppt.progress_bar(i+1-nmonths* month_ndays+1,len(factor)-nmonths* month_ndays+1)
                factor.iloc[i,:] = data.iloc[i-nmonths* month_ndays+1:i,:].mean(axis=0)*nmonths* month_ndays
        elif freq == "Y":
            nmonths = year_nmonths
            for i in range(nmonths* month_ndays-1,len(factor)):
                ppt.progress_bar(i+1-nmonths* month_ndays+1,len(factor)-nmonths* month_ndays+1)
                factor.iloc[i,:] = data.iloc[i-nmonths* month_ndays+1:i,:].mean(axis=0)*nmonths* month_ndays
        
        factor = pd.DataFrame(factor,dtype=float)
        # 计算自然对数
        if logflag:
            factor[factor==0] = np.nan
            factor = np.log(factor/nmonths)
        
        # # 填充空值
        # factor = ppt.fill_missing(factor,"fixed")
        
        return factor
    
    
    # ================================================================================
    # 盈利因子
    # name          指标名称，["ETOP","CETOP"]
    # category      选择市值类型，["总市值","流通市值"]
    # ================================================================================
    def EarningYieldFactorExpo(self,name,category,Date):
        
        # 选择市值类型
        data = eval(f"self.{self.name_dict[category]}")
        
        # 调整行列索引
        data = data.loc[self.start_date:self.end_date] \
            .reindex(columns=self.stock_info.index)
        
        #重新赋值，防止引用传递           
        date = Date.copy()
        
        # 数据文件位置
        path = "data/quarterly/stock"
        
        # 净利润/市值
        if name == "ETOP":
            numer = pd.read_pickle(f"{path}/NET_PROFIT_EXCL_MIN_INT_INC").loc[self.start_date:self.end_date].resample("Q").last()
        
        # net CFO/市值
        elif name == "CETOP":
            numer = pd.read_pickle(f"{path}/NET_CASH_FLOWS_OPER_ACT").loc[self.start_date:self.end_date].resample("Q").last()
        
        # 季频数据转换为TTM
        numer = ppt.transform_ttm_in_wind_way(numer)
        
        for i,istock in enumerate(date.columns):
            ppt.progress_bar(i+1,len(date.columns))
            date.loc[:,istock] = date.loc[:,istock].apply(lambda x: np.nan if pd.isnull(x) else numer.loc[x,istock])
        
        # 向后填充
        E = date.fillna(method='ffill')
        E = E.loc[self.start_date:self.end_date]
        

        # 计算因子
        factor = E/data
        
        # # 填充空值
        # factor = ppt.fill_missing(factor,"fixed")
        
        return factor
    
    
    # ================================================================================
    # 成长因子
    # name          指标名称，["EGRO","SGRO"]
    # ================================================================================
    def GrowthFactorExpo(self,name,Date):
        
        # 数据文件位置
        path = "data/quarterly/stock"
        
        #重新赋值，防止引用传递           
        date = Date.copy()
        
        # 净利润
        if name == "EGRO":
            data = pd.read_pickle(f"{path}/NET_PROFIT_EXCL_MIN_INT_INC").resample("Y").last()
        
        # 营业收入
        elif name == "SGRO":
            data = pd.read_pickle(f"{path}/TOT_OPER_REV").resample("Y").last()
        
        # 计算过去5年复合增长率
        factor = ppt.cal_cagr(data,5).loc[self.start_date:self.end_date]
        
        for i,istock in enumerate(date.columns):
            ppt.progress_bar(i+1,len(date.columns))
            date.loc[:,istock] = date.loc[:,istock].apply(lambda x: np.nan if pd.isnull(x) else factor.loc[x,istock])
        
        # 向后填充
        E = date.fillna(method='ffill')
        E = E.loc[self.start_date:self.end_date]
        
        # # 填充空值
        # E = ppt.fill_missing(E,"fixed")
        
        return E
    
    
    # ================================================================================
    # 杠杆因子
    # name          指标名称，["MLEV","DTOA","BLEV"]
    # category      选择市值类型，["总市值","流通市值"]
    # ================================================================================
    def LeverageFactorExpo(self,name,category,Date):
        
        # 选择市值类型
        data = eval(f"self.{self.name_dict[category]}")
        
        # 调整行列索引
        data = data.loc[self.start_date:self.end_date] \
            .reindex(columns=self.stock_info.index)
            
        
        # 数据文件位置
        path = "data/quarterly/stock"
        
                
        # 长期负债/市值
        if name == "MLEV":
            
            #重新赋值，防止引用传递           
            date = Date.copy()
            numer = pd.read_pickle(f"{path}/TOT_LIAB").loc[self.start_date:self.end_date].resample("Y").last() - \
                    pd.read_pickle(f"{path}/TOT_CUR_LIAB").loc[self.start_date:self.end_date].resample("Y").last()
            denom = data
                   
            for i,istock in enumerate(date.columns):
                ppt.progress_bar(i+1,len(date.columns))
                date.loc[:,istock] = date.loc[:,istock].apply(lambda x: np.nan if pd.isnull(x) else numer.loc[x,istock])
            
            # 向后填充
            LD = date.fillna(method='ffill')
            LD = LD.loc[self.start_date:self.end_date]
            
            factor = 1+LD/denom
        
        # 总负债/总资产
        elif name == "DTOA":
            #重新赋值，防止引用传递           
            date1 = Date.copy()
            date2 = Date.copy()
            numer = pd.read_pickle(f"{path}/TOT_LIAB").loc[self.start_date:self.end_date].resample("Y").last()
            denom = pd.read_pickle(f"{path}/TOT_ASSETS").loc[self.start_date:self.end_date].resample("Y").last()
            
            for i,istock in enumerate(date1.columns):
                ppt.progress_bar(i+1,len(date1.columns))
                date1.loc[:,istock] = date1.loc[:,istock].apply(lambda x: np.nan if pd.isnull(x) else numer.loc[x,istock])
            
            # 向后填充
            TD = date1.fillna(method='ffill')
            TD = TD.loc[self.start_date:self.end_date]
            
            for i,istock in enumerate(date1.columns):
                ppt.progress_bar(i+1,len(date1.columns))
                date2.loc[:,istock] = date2.loc[:,istock].apply(lambda x: np.nan if pd.isnull(x) else denom.loc[x,istock])
            # 向后填充
            TA = date2.fillna(method='ffill')
            TA = TA.loc[self.start_date:self.end_date]

            factor = TD/TA
        
        # 长期负债/净资产
            #重新赋值，防止引用传递
        elif name == "BLEV":
            date1 = Date.copy()
            date2 = Date.copy()
            numer = pd.read_pickle(f"{path}/TOT_LIAB").loc[self.start_date:self.end_date].resample("Y").last() -\
                    pd.read_pickle(f"{path}/TOT_CUR_LIAB").loc[self.start_date:self.end_date].resample("Y").last()
            denom = pd.read_pickle(f"{path}/TOT_SHRHLDR_EQY_EXCL_MIN_INT").loc[self.start_date:self.end_date].resample("Y").last()
            
            for i,istock in enumerate(date1.columns):
                ppt.progress_bar(i+1,len(date1.columns))
                date1.loc[:,istock] = date1.loc[:,istock].apply(lambda x: np.nan if pd.isnull(x) else numer.loc[x,istock])
            
            # 向后填充
            LD = date1.fillna(method='ffill')
            LD = LD.loc[self.start_date:self.end_date]
            
            for i,istock in enumerate(date1.columns):
                ppt.progress_bar(i+1,len(date1.columns))
                date2.loc[:,istock] = date2.loc[:,istock].apply(lambda x: np.nan if pd.isnull(x) else denom.loc[x,istock])
            
            # 向后填充
            BE = date2.fillna(method='ffill')
            BE = BE.loc[self.start_date:self.end_date]
           
            factor = 1+LD/BE
        
        # # 填充空值
        # factor = ppt.fill_missing(factor,"fixed")
        
        return factor
    
    # 获取日频索引
    def fill_QorY(self,datetype):
        
        # quarterLoc = self.quarterLoc.loc[self.start_date:self.end_date].resample(datetype).last()
        
        # columns = self.total_mkv.loc[self.start_date:] \
        #         .reindex(columns=self.stock_info.index).columns
        # index = self.total_mkv.loc[self.start_date:] \
        #         .reindex(columns=self.stock_info.index).index   
                
        # data = pd.DataFrame(columns=columns,index=index).loc[self.start_date:self.end_date]
        
        # date = ppt.fill_Q(quarterLoc,data)
        
        # 日期数据
        quarterLoc = model.quarterLoc.loc[model.start_date:model.end_date].resample(datetype).last()
        
        # str形式的日期序列转换为datetime形式
        quarterLoc = quarterLoc.applymap(lambda x : (np.nan if pd.isnull(x) else pd.to_datetime(x)))
        
        # 分成不同的series, 进行map处理
        series_list = [quarterLoc.loc[:,i] for i in quarterLoc.columns]
        
        # 读取计算结果
        df_date = pd.concat(map(ppt.get_pct, series_list), axis=1)
        
        # 按日频展开
        df_date = df_date.resample('D').asfreq() 
        
        return df_date
    
    # ================================================================================
    # 风格因子暴露正交化、标准化
    # styleExpo     风格因子暴露集合(list)
    # category      选择市值类型，["总市值","流通市值"]
    # issqrt        是否对市值开根
    # ================================================================================
    def FactorProcess(self,styleExpo,category,issqrt=False):
        # 选择市值类型
        mkv = eval(f"self.{self.name_dict[category]}")
        
        # 调整行列索引
        mkv = mkv.loc[self.start_date:self.end_date] \
            .reindex(columns=self.stock_info.index)
        notnull = mkv.notnull()
              
        # 百分位数去极值
        mkv = ppt.winsorize(mkv,"quantile",axis=1,lower=0,upper=0.95)
        
        # 因子标准化处理
        for i in styleExpo.keys():
            
            # 暂时不处理Liquidity、ResidVola
            if i == 'Liquidity' or i == 'ResidVola':
                continue
            #去掉无效日期
            styleExpo[i].dropna(how='all',inplace=True)
            Mkv = mkv.reindex(index=styleExpo[i].index)
            
            # 中位数去极值
            styleExpo[i] = pd.DataFrame(styleExpo[i],dtype=float)
            styleExpo[i] = ppt.winsorize(styleExpo[i],"median-MAD",axis=1,multiplier=3)
            
            # 填充空值
            cate_df = self.indus1_belong.reindex(columns=self.stock_info.index,index=styleExpo[i].index)
            styleExpo[i] = ppt.fill_missing(styleExpo[i],"category_mean",axis=1,cate_df=cate_df,thres=0)
            # styleExpo[i] = ppt.fill_missing(styleExpo[i],"fixed")
            
            # 去掉无效数据
            
            styleExpo[i] = styleExpo[i][notnull]
            
            # 根据参数设置确定是否对市值开根号
            if issqrt:
                Mkv = np.sqrt(Mkv)
            
            # 把无效股票的市值置为空值
            Mkv = Mkv.reindex(columns=styleExpo[i].columns)[styleExpo[i].notnull()]
            
            # 使用加权均值进行标准化
            styleExpo_mean = (styleExpo[i]*Mkv).sum(axis=1) / Mkv.sum(axis=1).replace(0,np.nan)
            styleExpo[i] = (styleExpo[i]-np.repeat(np.expand_dims(styleExpo_mean,1),\
                            len(styleExpo[i].columns),1))/np.repeat(np.expand_dims\
                            (styleExpo[i].std(axis=1).replace(0,np.nan),1),len(styleExpo[i].columns),1) 
        
        # 波动率因子对规模和Beta因子正交化
        styleExpo['Beta'].dropna(how='all',inplace=True)
        dataX = styleExpo['Size'].reindex(index=styleExpo['Beta'].index)
        dataY = styleExpo['ResidVola'].reindex(index=styleExpo['Beta'].index)
        
        factor = pd.DataFrame(index=dataY.index,columns=dataY.columns)
        
        length = factor.shape[0]
        for i,ind in enumerate(factor.index):
            
            # 进度条
            ppt.progress_bar(i+1,length)
            
            # 获取截面数据
            y = dataY.loc[ind]
            x = dataX.loc[ind]
            w = mkv.loc[ind]
            
            # 获取截面上的有效索引
            valid_locs = y.notnull() & x.notnull() & w.notnull()
            if sum(valid_locs) == 0:
                continue
            _,_,resid = ppt.weighted_regression( \
                y[valid_locs],x[valid_locs],weight=w[valid_locs])
            factor.loc[ind,valid_locs] = resid[:,0]
        
        # 填充空值
        #dataY = ppt.fill_missing(factor,"fixed") 
        dataY = factor
        dataX = styleExpo['Beta']
        
        factor = pd.DataFrame(index=dataY.index,columns=dataY.columns)
        
        length = factor.shape[0]
        for i,ind in enumerate(factor.index):
            
            # 进度条
            ppt.progress_bar(i+1,length)
            
            # 获取截面数据
            y = dataY.loc[ind]
            x = dataX.loc[ind]
            w = mkv.loc[ind]
            
            # 获取截面上的有效索引
            valid_locs = y.notnull() & x.notnull() & w.notnull()
            if sum(valid_locs) == 0:
                continue
            _,_,resid = ppt.weighted_regression( \
                y[valid_locs],x[valid_locs],weight=w[valid_locs])
            factor.loc[ind,valid_locs] = resid[:,0]
        
        #styleExpo['ResidVola'] = ppt.fill_missing(factor,"fixed")
        styleExpo['ResidVola'] = pd.DataFrame(factor,dtype=np.float)
       
        # 流动性因子对规模因子正交化
        dataY = styleExpo['Liquidity']
        dataX = styleExpo['Size'].reindex(index=dataY.index)
       
        factor = pd.DataFrame(index=dataX.index,columns=dataX.columns)
        
        length = factor.shape[0]
        for i,ind in enumerate(factor.index):
            
            # 进度条
            ppt.progress_bar(i+1,length)
            
            # 获取截面数据
            y = dataY.loc[ind]
            x = dataX.loc[ind]
            w = mkv.loc[ind]
            
            # 获取截面上的有效索引
            valid_locs = y.notnull() & x.notnull() & w.notnull()
            if sum(valid_locs) == 0:
                continue
            _,_,resid = ppt.weighted_regression( \
                y[valid_locs],x[valid_locs],weight=w[valid_locs])
            factor.loc[ind,valid_locs] = resid[:,0]
        
        # 填充空值
        # styleExpo['Liquidity'] = ppt.fill_missing(factor,"fixed")
        styleExpo['Liquidity'] = pd.DataFrame(factor,dtype=np.float)
        
        # 因子标准化处理
        
        for i in styleExpo.keys():
            
            #处理Liquidity、ResidVola
            if i != 'Liquidity' and i != 'ResidVola':
                continue
            
            #去掉无效日期
            styleExpo[i].dropna(how='all',inplace=True)
            Mkv = mkv.reindex(index=styleExpo[i].index)
            
            # 中位数去极值
            styleExpo[i] = pd.DataFrame(styleExpo[i],dtype=float)
            styleExpo[i] = ppt.winsorize(styleExpo[i],"median-MAD",axis=1,multiplier=3)
            
            # 填充空值
            cate_df = self.indus1_belong.reindex(columns=self.stock_info.index,index=styleExpo[i].index)
            styleExpo[i] = ppt.fill_missing(styleExpo[i],"category_mean",axis=1,cate_df=cate_df,thres=0)
            # styleExpo[i] = ppt.fill_missing(styleExpo[i],"fixed")
            
            # 去掉无效数据
            
            styleExpo[i] = styleExpo[i][notnull]
            
            # 根据参数设置确定是否对市值开根号
            if issqrt:
                Mkv = np.sqrt(Mkv)
            
            # 把无效股票的市值置为空值
            Mkv = Mkv.reindex(columns=styleExpo[i].columns)[styleExpo[i].notnull()]
            
            # 使用加权均值进行标准化
            styleExpo_mean = (styleExpo[i]*Mkv).sum(axis=1) / Mkv.sum(axis=1).replace(0,np.nan)
            styleExpo[i] = (styleExpo[i]-np.repeat(np.expand_dims(styleExpo_mean,1),\
                            len(styleExpo[i].columns),1))/np.repeat(np.expand_dims\
                            (styleExpo[i].std(axis=1).replace(0,np.nan),1),len(styleExpo[i].columns),1) 
        
        return styleExpo    
 
        
    # ================================================================================
    # 因子收益率和因子暴露（返回增量更新部分）
    # styleExpo     风格因子暴露集合(list)
    # category      选择市值类型，["总市值","流通市值"]
    # issqrt        是否对市值开根
    # Num           各类因子数目
    # ================================================================================
    def FactorReturn(self,styleExpo,category,Num):
        # 统一日期索引
        for i in styleExpo.keys():
            styleExpo[i] = styleExpo[i].reindex(columns=self.stock_info.index,index=self.index_close.index)\
            .loc[self.start_date:self.end_date]
        
        # 行业暴露
        indusExpo = self.indus1_belong.reindex(columns=self.stock_info.index,index=self.index_close.index)\
            .loc[self.start_date:self.end_date]
    
        # 各类因子数目
        styleNum = Num['styleNum']
        indusNum = Num['indusNum']
        countryNum = Num['countryNum']
        factorNum = styleNum+indusNum+countryNum
    
        # 个股收益率序列
        stockReturn = self.daily_ret.reindex(columns=self.stock_info.index,index=self.index_close.index)\
            .loc[self.start_date:self.end_date]
        
        # 选择市值类型
        mkv = eval(f"self.{self.name_dict[category]}")
        
        # 调整行列索引
        mkv = mkv.reindex(columns=self.stock_info.index,index=self.index_close.index)\
            .loc[self.start_date:self.end_date]
    
        # 百分位数去极值
        mkv = ppt.winsorize(mkv,"quantile",axis=1,lower=0,upper=0.95)
        
        # 初始化结果
        factorExpo = {}
        fac_names = ['Size', 'Beta', 'Momentum', 'ResidVola', 'Nonlinear', 'BooktoPrice',
       'Liquidity', 'EarningYield', 'Growth', 'Leverage', '交通运输', '传媒', '农林牧渔',
       '医药', '商贸零售', '国防军工', '基础化工', '家电', '建材', '建筑', '房地产', '有色金属', '机械',
       '汽车', '消费者服务', '煤炭', '电力及公用事业', '电力设备及新能源', '电子', '石油石化', '纺织服装', '综合',
       '综合金融', '计算机', '轻工制造', '通信', '钢铁', '银行', '非银行金融', '食品饮料', '国家']
        factorReturn = pd.DataFrame(index=indusExpo.index,columns=fac_names)  

        for i,iday in enumerate(stockReturn.index):
            
            # 进度条
            ppt.progress_bar(i+1,len(stockReturn.index))
            
            # 获取截面上的风格因子暴露
            panelStyleExpo = pd.concat([styleExpo['Size'].loc[iday,:],styleExpo['Beta'].loc[iday,:],styleExpo['Momentum'].loc[iday,:]
                                       ,styleExpo['ResidVola'].loc[iday,:],styleExpo['NonLinear'].loc[iday,:],styleExpo['BooktoPrice'].loc[iday,:]
                                       ,styleExpo['Liquidity'].loc[iday,:],styleExpo['EarningYield'].loc[iday,:],styleExpo['Growth'].loc[iday,:]
                                       ,styleExpo['Leverage'].loc[iday,:]],axis=1).T
            panelStyleExpo.index = ['Size','Beta','Momentum','ResidVola','Nonlinear','BooktoPrice','Liquidity','EarningYield','Growth','Leverage']
            
            # 获取行业因子哑变量矩阵
            panelIndusExpo = pd.get_dummies(indusExpo.loc[iday,:]).T.iloc[1:indusNum+1,:]
            
            # 国家因子
            panelCountryExpo = pd.DataFrame(columns=panelIndusExpo.columns,index=['国家'])
            panelCountryExpo.fillna(1,inplace=True)
            
            # 保存因子暴露结果
            total_factor = pd.concat([panelStyleExpo,panelIndusExpo,panelCountryExpo])
            
            
            # 前期行业个数只有29个时，会导致边界读取错误
            # 这里先做条件判断，未来会把行业底层归属数据
            # 替换成无加工版本
            if len(total_factor.index) < factorNum:
                factorExpo[iday] = total_factor.reindex(index=factorReturn.columns,fill_value=0)
            else:
                factorExpo[iday] = total_factor
            # 最新一天的因子暴露已经无法计算收益率
            if iday ==  stockReturn.index[-1]:
               continue
           
            # 获取截面股票收益率（注意是用T日因子对T+1日收益率回归）
            panelReturn = pd.DataFrame(stockReturn.iloc[1+i,:]).T
   
            # 获取有完整数据的个股（风格暴露、行业暴露、股票收益率均为有效值）
            lineNoNan = ~(pd.concat([factorExpo[iday],panelReturn]).isnull().any())
            if sum(lineNoNan) == 0:
                continue
            
            # 获取有效的回归自变量、因变量、权重
            Y = panelReturn.loc[:,lineNoNan].values
            X = pd.concat([panelStyleExpo,panelIndusExpo]).loc[:,lineNoNan].values
            w = mkv.loc[iday,lineNoNan]
            
            # 回归系数估计
            beta = np.linalg.inv(X @ np.diag(w) @ X.T) @ X @ np.diag(w) @ Y.T
            
            # 风格因子收益率直接取回归结果
            styleReturn = beta[0:styleNum]
            
            # 国家因子收益率是行业因子收益率的市值加权
            indusWeight = np.expand_dims(w @ panelIndusExpo.loc[:,lineNoNan].T,1)
            indusReturn = beta[styleNum:]
            countryReturn = np.nansum(indusReturn * indusWeight)/\
                            np.nansum((~np.isnan(indusReturn)) * indusWeight)
            
            # 行业因子收益率是在原来基础上减去国家因子收益率
            indusReturn = indusReturn - countryReturn
            total = pd.DataFrame(np.vstack((styleReturn,indusReturn,countryReturn)).T
                                 ,columns=total_factor.index).reindex(columns=factorExpo[iday].index)
            factorReturn.iloc[i+1,:] = total.values
            
        return factorReturn,factorExpo
    
    
    # ================================================================================
    # 计算残差收益率
    # factorExpo    因子暴露
    # factorReturn  因子收益
    # ================================================================================    
    def SpecialReturn(self,factorExpo,factorReturn):
       
            # 个股收益率序列
            stockReturn = self.daily_ret.reindex(columns=self.stock_info.index,index=self.index_close.index)\
                        .loc[self.start_date:self.end_date]
    
            # 初始化结果
            specialReturn = pd.DataFrame(columns=stockReturn.columns,index=stockReturn.index)
    
            # 特异性收益率计算
            for i,iday in enumerate(stockReturn.index):
                
                # 进度条
                ppt.progress_bar(i+1,len(stockReturn.index))
                
                # 最后一次无法计算收益率
                if iday ==  stockReturn.index[-1]:
                    continue
                
                # 因子暴露
                panelExpo = factorExpo[iday]
                
                # 因子收益
                panelFactorReturn = factorReturn.iloc[i+1,:]
                panelFactorReturn = panelFactorReturn.fillna(value=0)
                
                # 计算残差收益率
                specialReturn.iloc[i+1,:] = (stockReturn.iloc[i+1,:] - np.expand_dims(panelFactorReturn,1).T @ panelExpo).values
    
            return specialReturn
            
if __name__ == "__main__":
    
    arg = {"指标开始日期":"2016-01-01",
           "指标结束日期":"2019-12-31",
           "基准指数代码":"000985.CSI"}
    model = gen_factor(arg)
    
    date_Q = model.fill_QorY("Q")
    date_Y = model.fill_QorY("Y")
    
    # date_Q.to_excel("date_Q.xlsx")
    # date_Y.to_excel("date_Y.xlsx")
    
    # date_Q = pd.read_excel("date_Q.xlsx",index_col=0)
    # date_Y = pd.read_excel("date_Y.xlsx",index_col=0)
    
    # ===========================================================
    # 生成因子暴露
    # ===========================================================
    
    # 行业因子
    INDUS = model.IndusFactorExpo()
    
    # 规模因子
    LNCAP = model.SizeFactorExpo("总市值")
    Size = LNCAP*1
    
    # 动量因子
    params = {"窗口天数":504,
              "半衰期长度":126}
    RSTR = model.MomentumFactorExpo(params["窗口天数"],params["半衰期长度"])
    Momentum = RSTR*1
    
    # Beta因子
    params = {"窗口天数":252,
              "半衰期长度":63}
    _,BETA,HSIGMA = model.BetaFactorExpo(params["窗口天数"],params["半衰期长度"])
    Beta = BETA*1
    
    # 波动率因子
    params = {"窗口天数":252,
              "半衰期长度":63,
              "滚动月份数":12}
    DASTD = model.ResidVolaFactorExpo("DASTD",params["窗口天数"],params["滚动月份数"],params["半衰期长度"])###没找到
    CMRA = model.ResidVolaFactorExpo("CMRA",params["窗口天数"],params["滚动月份数"],params["半衰期长度"])###相对大小差不多，绝对大小差2倍
    ResidVola = HSIGMA*0.1+CMRA*0.16+DASTD*0.74
    
    # 非线性因子
    NLSIZE = model.NonLinearFactorExpo("流通市值",issqrt=True)
    NonLinear = NLSIZE*1
    
    # 市净率因子
    BTOP = model.BooktoPriceFactorExpo("总市值",date_Y)
    BooktoPrice = BTOP*1
    
    # 流动性因子
    STOM = model.LiquidityFactorExpo("M")
    STOQ = model.LiquidityFactorExpo("Q")
    STOA = model.LiquidityFactorExpo("Y")
    Liquidity = STOM*0.35+STOQ*0.35+STOA*0.3
    
    # 盈利因子
    ETOP = model.EarningYieldFactorExpo("ETOP","总市值",date_Q)###
    CETOP = model.EarningYieldFactorExpo("CETOP","总市值",date_Q)###
    EarningYield = ETOP*0.66+CETOP*0.34
    
    # 成长因子 净利润/营业收入数据不同
    EGRO = model.GrowthFactorExpo("EGRO",date_Y)
    SGRO = model.GrowthFactorExpo("SGRO",date_Y)
    Growth = EGRO*0.34+SGRO*0.66
    
    # 杠杆因子 净资产数据不同
    MLEV = model.LeverageFactorExpo("MLEV","总市值",date_Y)###
    DTOA = model.LeverageFactorExpo("DTOA","总市值",date_Y)###
    BLEV = model.LeverageFactorExpo("BLEV","总市值",date_Y)###
    Leverage = MLEV*0.38+DTOA*0.35+BLEV*0.27
    
    
    # 因子暴露正交化
    styleExpo = {'Size':Size,
                  'Momentum':Momentum,
                  'Beta':Beta,
                  'ResidVola':ResidVola,
                  'NonLinear':NonLinear,
                  'BooktoPrice':BooktoPrice,
                  'Liquidity':Liquidity,
                  'EarningYield':EarningYield,
                  'Growth':Growth,
                  'Leverage':Leverage}
        
    # f_save = open('data/step1/styleExpo_before.pkl', 'wb')
    # pickle.dump(styleExpo, f_save)
    # f_save.close()
    
    # f_read = open('data/step1/styleExpo_before.pkl', 'rb')
    # styleExpo = pickle.load(f_read)
    # f_read.close()
      
    styleExpo = model.FactorProcess(styleExpo,'流通市值',issqrt=True)
    
    # f_save = open('data/step1/styleExpo.pkl', 'wb')
    # pickle.dump(styleExpo, f_save)
    # f_save.close()
    
    # f_read = open('data/step1/styleExpo.pkl', 'rb')
    # styleExpo = pickle.load(f_read)
    # f_read.close()
    
    # =============================================================
    # 因子收益率、残差收益率计算
    # =============================================================
    
    Num = {'styleNum':10,
            'indusNum':30,
            'countryNum':1}
    # 计算因子收益率和因子暴露
    factorReturn,factorExpo = model.FactorReturn(styleExpo,'流通市值',Num)
    
    # f_save = open('data/step1/factorReturn.pkl', 'wb')
    # pickle.dump(factorReturn, f_save)
    # f_save.close()
    
    # f_save = open('data/step1/factorExpo.pkl', 'wb')
    # pickle.dump(factorExpo, f_save)
    # f_save.close()
    
    # f_read = open('data/step1/factorExpo.pkl', 'rb')
    # factorExpo = pickle.load(f_read)
    # f_read.close()
    
    # f_read = open('data/step1/factorReturn.pkl', 'rb')
    # factorReturn = pickle.load(f_read)
    # f_read.close()
    
    # 计算残差收益率
    specialReturn = model.SpecialReturn(factorExpo,factorReturn)
    
    # f_save = open('data/step1/specialReturn.pkl', 'wb')
    # pickle.dump(specialReturn, f_save)
    # f_save.close()
    
    # f_read = open('data/step1/specialReturn.pkl', 'rb')
    # specialReturn = pickle.load(f_read)
    # f_read.close()
    
    # writer = pd.ExcelWriter('结果对比2.xlsx')
    # factorExpo[pd.to_datetime("2019-12-02")].to_excel(writer,sheet_name='因子暴露')
    # factorReturn.to_excel(writer,sheet_name='因子收益率')
    # specialReturn.to_excel(writer,sheet_name='残差收益率')
    # writer.save()
    
    
    
    
    
    
    
    
    