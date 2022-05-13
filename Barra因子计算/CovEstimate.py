# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:46:44 2021

@author: Administrator
"""

import numpy as np
import pandas as pd
from utils import PreprocessingTools as ppt

class gen_factor:
    
    def __init__(self):
        
        # 因子暴露
        self.factorExpo = pd.read_pickle("data/step1/factorExpo.pkl")
        
        # 因子收益
        self.factorReturn = pd.read_pickle("data/step1/factorReturn.pkl")
        
        # 残差收益
        self.specialReturn = pd.read_pickle("data/step1/specialReturn.pkl")
        
    # ================================================================================
    # 因子协方差估计：Newey-West调整
    # params           参数
    # ================================================================================
    def FactorCovNeweyWest(self,params):
        
        # 因子收益
        factorReturn = self.factorReturn
        
        # 参数准备
        tBegin = pd.to_datetime(params['起始日期'])
        timeWin = params['窗口天数']
        halfLife = params['半衰期长度']
        dayNumOfMonth = params['每月的交易日数']
        D = params['自相关项滞后期数']
        
        # 半衰期权重序列
        weightList = 0.5**(np.arange(timeWin,0,-1)/halfLife)
        
        # 初始化结果
        factorCov = {}
        
        # 遍历每个截面
        for i,iday in enumerate(factorReturn.index[:630]):
            
            # 判断起始日期
            if iday < tBegin:
                factorCov[iday] = pd.DataFrame(columns=factorReturn.columns,index=factorReturn.columns)
                continue
            
            # 获取窗口期样本
            data = factorReturn.iloc[i-timeWin+1:i+1,:]
            if data.count().sum() < 5:
                data.loc[:,:] = np.nan

            # 计算因子协方差
            weight = weightList / sum(weightList)
            data = (data - np.tile(np.nansum(data.T*weight,axis=1),[timeWin,1])).values
            FNW = data.T @ np.diag(weight) @ data

            # 考虑自相关项
            for q in range(D):
                k = 1- (q+1)/(D+1)
                sumw = sum(weightList[q+1:])
                weight = weightList[q+1:] / sumw
                dataleft = data[q+1:,:].T @ np.diag(weight) @ data[q+1:,:]
                dataright = data[:-q-1,:].T @ np.diag(weight) @ data[:-q-1,:]
                FNW = FNW + k * (dataleft + dataright)
                
            FNW = pd.DataFrame(FNW,columns=factorReturn.columns,index=factorReturn.columns)
            factorCov[iday] = dayNumOfMonth * FNW
        
        return factorCov

    # ================================================================================
    # 因子协方差估计：特征值调整
    # factorCovNW      NW调整后的因子协方差矩阵
    # params           参数
    # ================================================================================
    def FactorCovEigenAdjust(self,paramSet,factorCovNW):
        
        # 参数设置
        MCS = params['蒙特卡洛模拟次数']                     
        A = params['特征值调整系数']      
        timeWin = params['窗口天数']
        
        # 初始化结果
        factorCovEigenAdj = {}
        #factorCovEigenAdjGamma = pd.DataFrame(columns=self.factorReturn.columns,index=self.factorReturn)
        
        # 循环计算所有截面的数据
        for i,iday in enumerate(self.factorReturn.index[:630]):
            
            ppt.progress_bar(i+1,len(self.factorReturn.index[:630]))
             
            # NAN值处理：由于行业个数出现过变更，较早之前的数据有一个因子维度有缺失
            tempFnw =  factorCovNW[iday]
            if tempFnw.isnull().all().all():
                continue
            Fnw = tempFnw.dropna(how='all',axis=0).dropna(how='all',axis=1)
            
            # 蒙特卡洛模拟：计算模拟风险偏差
            # 计算特征值特征向量，特征值升序排列
            D0, U0 = np.linalg.eig(np.array(Fnw,dtype=np.float))
            tempRM = U0*np.sqrt(D0)
            lamb = np.zeros((1,len(Fnw)))
            for iterMCS in range(MCS):
                rd = np.random.RandomState(iterMCS)
                rm = np.dot(tempRM , rd.random((len(Fnw),timeWin)))
                Fm = np.cov(rm)
                Dm, Um = np.linalg.eig(Fm)
                DmReal = np.array(np.diag(np.dot(np.dot(Um.T,Fnw),Um)),dtype = np.float)
                lamb=lamb+DmReal/Dm
            
            lamb = np.sqrt(lamb/2)
            
            # 协方差矩阵：特征值调整
            gamma = np.dot(A ,(lamb-1)) + 1
            D0Real = D0 * (gamma**2)
            Feigen = np.dot(np.dot(U0,D0Real.T).T,U0)
        
            factorCovEigenAdj[iday] = Feigen
            
        return factorCovEigenAdj
        

    # ================================================================================
    # 因子协方差估计：特征值调整
    # factorCovEigenAdj      特征值调整后的因子协方差矩阵
    # params           参数
    # ================================================================================
    def FactorCovVolRegAdj(self,paramSet,factorCovEigenAdj):
       
        # 参数设置
        timeWin = params['窗口天数']
        halfLife = params['半衰期']
        
        # 生成因子未来一个月的对数收益率        
        logReturnDay = np.log(self.factorReturn.fillna(0)+1).reindex(list(factorCovEigenAdj.keys())[:-21])
        logReturnMonth = pd.DataFrame(columns=logReturnDay.columns,index=list(factorCovEigenAdj.keys())[:-21])
        
        for i,iday in enumerate(list(factorCovEigenAdj.keys())[:-21]):
            logReturnMonth.loc[iday,:] = np.sum(logReturnDay.iloc[i:(i+21-1),:],0)
        returnMonth = np.exp(np.array(logReturnMonth,dtype=float)) - 1
        
        
        singleFactorEigAdj = pd.DataFrame(columns=logReturnDay.columns,index=list(factorCovEigenAdj.keys()))
        # 获取单个因子特征值调整后的波动率
        for i,iday in enumerate(list(factorCovEigenAdj.keys())):
            if len(factorCovEigenAdj[iday]) == 41:
                singleFactorEigAdj.loc[iday,:] = factorCovEigenAdj[iday]
            else:
                singleFactorEigAdj.loc[iday,:'综合'] = factorCovEigenAdj[iday][0][:32]
                singleFactorEigAdj.loc[iday,"计算机":] = factorCovEigenAdj[iday][0][32:]

        
        # 计算所有因子的总偏误统计量
        FktToSIGMAkt = returnMonth**2 / np.array(singleFactorEigAdj.reindex(index=logReturnMonth.index),dtype=float)             
        BFt = np.nanmean(FktToSIGMAkt,1)  
                                                  
        weight = 0.5**(np.arange(timeWin,0,-1)/halfLife)
        
        factorCovVolRegAdj={}
        for i in range(timeWin-1,len(BFt)):
            # 计算波动率调整系数
            data = BFt[i-timeWin+1:i]
            lambdaF = np.nansum(data*weight) / np.sum(weight[~np.isnan(data)])
            
            # 进行波动率调整
            iday = list(factorCovEigenAdj.keys())[i]
            factorCovVolRegAdj[iday] = lambdaF * factorCovEigenAdj[iday]     
        
        return factorCovVolRegAdj


if __name__ == "__main__":
    
    model = gen_factor()
    
    # ================================================================================
    # 因子协方差估计
    # ================================================================================
    
    # 因子协方差估计：Newey-West调整
    params={'起始日期':"2017-12-31",
            '窗口天数':100,
            '半衰期长度':90,
            '每月的交易日数':21,
            '自相关项滞后期数':2}
    factorCovNW = model.FactorCovNeweyWest(params)
    
    # 因子协方差估计：特征值调整
    params={'窗口天数':100,
            '蒙特卡洛模拟次数':3000,
            '特征值调整系数':1.5}
    factorCovEigenAdj = model.FactorCovEigenAdjust(params,factorCovNW)

    
    # 因子协方差估计：波动率偏误调整
    params={'窗口天数':252,
            '半衰期':42}    
    factorCovVolRegAdj = model.FactorCovVolRegAdj(params,factorCovEigenAdj)











        
        