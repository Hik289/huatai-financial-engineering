# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 09:36:04 2021

@author: kipcrk290
"""
import numpy as np
import pandas as pd

# ================================================================================
# 进度条
# ================================================================================
def progress_bar(i,n):
    pct = i/n*100
    p1 = int(pct/2)*"#"
    p2 = (50-int(pct/2))*"-"
    print("\r %6.2f%% | [%s%s] | %d/%d" % (pct,p1,p2,i,n),end="")
    if i == n:
        print()


# ================================================================================
# 计算加权平均值和中心矩
# df            DataFrame
# window        滚动天数，int
# n             计算n阶加权中心矩
# weight        权重，list, Series或1d-array
# mean_method   选择计算平均值或中位数，["mean","median"]
# fillna        是否把nan作为0进行处理，bool
# pbar          是否输出进度条
# ================================================================================
def rolling_weighted_moment(df,window,n=2,weight=None,mean_method="mean",fillna=False,pbar=True):
    
    # df = pd.DataFrame(df,dtype=float)
    # 判断是否加权计算
    if weight is None:
        weight = np.ones(window)
    
    # 选择是否填充nan
    if not fillna:
        arr = df.values
    else:
        arr = df.fillna(0).values
    
    # 生成滚动窗口
    rolling_windows = gen_rolling_windows(arr,window)
    
    # 权重调整为2d-array
    if len(weight.shape) == 1:
        weight = np.expand_dims(weight,1)
    
    # 初始化输出结果
    wm = np.empty([rolling_windows.shape[0],rolling_windows.shape[2]])
    warr = np.empty([rolling_windows.shape[0],rolling_windows.shape[2]])
    
    for i in range(rolling_windows.shape[0]):
        
        # 进度条
        if pbar:
            progress_bar(i+1,rolling_windows.shape[0])
        
        # 根据数据中的空值调整权重
        # a = rolling_windows[i].astype(np.float32)
        adj_weight = ~np.isnan(rolling_windows[i]) * weight
        wSum = np.nansum(adj_weight,axis=0)
        wSum[wSum==0] = np.nan
        
        # 计算当前截面加权后的序列
        weighted_seq = rolling_windows[i]*adj_weight
        #weighted_seq = weighted_seq.astype(np.float32)
        
        # 计算加权平均值/中位数
        if mean_method == "mean":
            wm[i] = np.nansum(weighted_seq,axis=0) / wSum
            
        elif mean_method == "median":
            
            # 计算加权后序列的中位数，获取最接近中位数的位置
            med = np.nanmedian(weighted_seq,axis=0)
            med_locs = np.argsort(np.abs(weighted_seq-med),axis=0)[0]
            
            # 在原始序列中提取对应位置的元素
            wm[i] = rolling_windows[i,med_locs] \
                [range(rolling_windows.shape[2]),range(rolling_windows.shape[2])]##取对角线
        
        # 计算n阶中心矩
        warr[i] = np.nansum(((rolling_windows[i]-wm[i])**n)*adj_weight,axis=0) / wSum
    
    # 调整行列索引
    res_mean = array2dataframe(wm,df)
    res_arr = array2dataframe(warr,df)
    
    return res_mean,res_arr


# ================================================================================
# 加权线性回归
# depVar        因变量，可以是list, array, DataFrame, Series
# indepVar      自变量
# weight        权重，None表示不加权
# intercept     选择是否添加常数项，bool
# ================================================================================
def weighted_regression(depVar,indepVar,weight=None,intercept=True):
    
    # 调整格式
    depVar = np.asarray(depVar)
    indepVar = np.asarray(indepVar)
    
    # 把自变量和因变量调整为2d-array
    if len(depVar.shape) == 1:
        depVar = np.expand_dims(depVar,1)
    if len(indepVar.shape) == 1:
        indepVar = np.expand_dims(indepVar,1)
    
    # 选择是否在自变量左侧添加常数列
    if intercept:
        const = np.ones([indepVar.shape[0],1])
        indepVar = np.concatenate([const,indepVar],axis=1)
    
    # 判断是否加权计算
    if weight is None:
        weight = np.ones(indepVar.shape[0])
    
    # 把权重调整为对角矩阵
    weight = np.diag(weight)
    
    # 计算回归系数和残差
    coef,resid = cal_resid(depVar,indepVar,weight)
    
    # 回归系数拆分
    if intercept:
        alpha = coef[0,0]
        beta = coef[1:,0]
    else:
        alpha = 0
        beta = coef[:,0]
    
    return alpha,beta,resid


# ================================================================================
# 滚动一元加权线性回归
# depVar        因变量，dataframe
# indepVar      自变量，dataframe
# window        滚动天数，int
# weight        权重，list, Series或1d-array
# fillna        是否把nan作为0进行处理，bool
# pbar          是否输出进度条
# ================================================================================
def rolling_weighted_regression(depVar,indepVar,window,weight=None,fillna=False,pbar=True):
    
    # 判断是否加权计算
    if weight is None:
        weight = np.ones(window)
    
    # 选择是否填充nan
    if not fillna:
        indepdf = indepVar.values
        depdf = depVar.values
    else:
        indepdf = indepdf.fillna(0).values
        depdf = depdf.fillna(0).values
    
    # 把自变量和因变量调整为2d-array
    if len(depdf.shape) == 1:
        depdf = np.expand_dims(depdf,1)
    if len(indepdf.shape) == 1:
        indepdf = np.expand_dims(indepdf,1)
    
    # 生成滚动窗口
    indep_rolling_windows = gen_rolling_windows(indepdf,window)
    dep_rolling_windows = gen_rolling_windows(depdf,window)
    
    # 权重调整为2d-array
    if len(weight.shape) == 1:
        weight = np.expand_dims(weight,1)
    
    # 初始化输出结果
    alpha = np.empty([dep_rolling_windows.shape[0],dep_rolling_windows.shape[2]])
    beta = alpha.copy()
    volresid = alpha.copy()
    
    for i in range(dep_rolling_windows.shape[0]):
        
        # 进度条
        if pbar:
            progress_bar(i+1,dep_rolling_windows.shape[0])
        
        # 根据数据中的空值调整权重
        adj_weight = ~(np.isnan(indep_rolling_windows[i]) | \
                       np.isnan(dep_rolling_windows[i])) * weight
        
        # 一元WLS公式
        elem_indep = indep_rolling_windows[i]
        elem_dep = dep_rolling_windows[i]
        
        wxSum = np.nansum(adj_weight*elem_indep,axis=0)
        wySum = np.nansum(adj_weight*elem_dep,axis=0)
        wxySum = np.nansum(adj_weight*elem_indep*elem_dep,axis=0)
        wxxSum = np.nansum(adj_weight*elem_indep**2,axis=0)
        wSum = np.nansum(adj_weight,axis=0)
        wSum[wSum==0] = np.nan
        
        denom = (wSum*wxxSum - wxSum**2)
        denom[denom==0] = np.nan
        
        # 计算回归系数
        beta[i] = (wSum*wxySum - wxSum*wySum) / denom
        alpha[i] = (wySum - wxSum * beta[i])/wSum
        
        # 计算残差波动率
        resid = elem_dep - beta[i]*elem_indep - alpha[i]
        _,s = rolling_weighted_moment(pd.DataFrame(resid),resid.shape[0],weight=weight,pbar=False)
        volresid[i] = np.sqrt(s.iloc[-1].values)
        
    # 调整行列索引
    res_alpha = array2dataframe(alpha,depVar)
    res_beta = array2dataframe(beta,depVar)
    res_volresid = array2dataframe(volresid,depVar)
    
    return res_alpha,res_beta,res_volresid


# ================================================================================
# 去极值
# df            DataFrame
# method        去极值的方法，["median-MAD","mean-MAD","median-std","mean-std","quantile","rank"]
# weight        权重，None表示不加权
# axis          计算方向，0为对列处理，1为对行处理
# multiplier    确定门限值的MAD/标准差的倍数（前四种方法适用）
# alpha         分位数参数（第五种方法适用）
# rank_method   排序方法（第六种方法适用）
# ================================================================================
def winsorize(df,method,weight=None,axis=0,multiplier=3,lower=0.025,upper=0.975,rank_method="average"):
    
    # 初始化
    df = df.copy()
    
    # 设定数据处理方向
    if axis == 1:
        df = df.T
    
    if method in ["median-MAD","mean-MAD","median-std","mean-std"]:
        
        # 字符串拆分
        method_list = method.split("-")
        
        # 计算中位数/平均值
        m,s = rolling_weighted_moment(df,df.shape[0],
            n=2,weight=weight,mean_method=method_list[0],pbar=False)
        m = m.iloc[-1]
        s = np.sqrt(s.iloc[-1])
        
        # 如果中位数/平均值为0，那么把对应列的0全部替换为nan，再重新计算中位数/平均值
        if (m==0).sum() > 0:
            df.loc[:,m==0] = df.loc[:,m==0].replace(0,np.nan)
            temp_m,temp_s = rolling_weighted_moment(df.loc[:,m==0],df.shape[0],
                n=2,weight=weight,mean_method=method_list[0],pbar=False)
            m[m==0] = temp_m.iloc[-1][m==0]
            s[m==0] = np.sqrt(temp_s.iloc[-1][m==0])
        
        # 计算MAD或标准差
        if method_list[1] == "MAD":
            # _,s = rolling_weighted_moment(df-m,df.shape[0],
            #     n=2,weight=weight,mean_method=method_list[0],pbar=False)
            # s = np.sqrt(s.iloc[-1])
            # radian = s
            mm,_ = rolling_weighted_moment(abs(df-m),df.shape[0],
                n=2,weight=weight,mean_method=method_list[0],pbar=False)
            mm = mm.iloc[-1]
            radian = mm
            
        elif method_list[1] == "std":
            radian = s
        
        # 把异常值替换为门限值
        upper = m+multiplier*radian
        lower = m-multiplier*radian
        df[df>=upper] = ((df>=upper)*upper)[df>=upper]
        df[df<=lower] = ((df<=lower)*lower)[df<=lower]
        
        # 填充空值
        df = df.fillna(0)
    
    elif method in ["quantile","rank"]:
        
        # 判断是否加权计算
        if weight is None:
            weight = np.ones(df.shape[0])
        
        # 把权重转化为2d-array
        weight = np.expand_dims(weight,1)
        
        # 计算加权序列
        adj_weight = (df.notnull()*weight).replace(0,np.nan)
        wSum = adj_weight.sum().replace(0,np.nan)
        weighted_df = (adj_weight*df)/wSum

        
        if method == "quantile":
            
            # 把异常值替换为门限值
            upper=weighted_df.quantile(q=upper)
            lower=weighted_df.quantile(q=lower)
            weighted_df[weighted_df>=upper] = ((weighted_df>=upper)*upper)[weighted_df>=upper]
            weighted_df[weighted_df<=lower] = ((weighted_df<=lower)*lower)[weighted_df<=lower]
            
            # 把加权序列还原到原始序列
            df = (weighted_df*adj_weight.sum()) / adj_weight
    
        elif method == "rank":
            
            # 把原始数据转换为排序值
            df = df.rank(method=rank_method)
    
    # 调整数据方向
    if axis == 1:
        df = df.T
    
    return df


# ================================================================================
# 填充缺失值
# df            DataFrame
# method        数据填充方式，["fixed","category_mean","category_median","ffill","bbill"]
# axis          计算方向，0为对列处理，1为对行处理
# fvalue        填充的数值（第一种方法适用）
# cate_df       按类别填充（第二和第三种方法适用）
# thres         最少有效数据个数（第二和第三种方法适用）
# ================================================================================
def fill_missing(df,method,axis=0,fvalue=0,cate_df=None,thres=20):
    
    # 初始化
    df = df.copy()
    
    # 设定数据处理方向
    if axis == 1:
        df = df.T
    
    if method == "fixed":
        df = df.fillna(fvalue)
    
    elif method in ["category_mean","category_median"]:
        
        # 调整数据方向
        if axis == 1:
            cate_df = cate_df.T
        
        # 字符串拆分
        method_list = method.split("_")
        
        for col in df.columns:
            
            # 计算填充值，如果有效数据过少，直接填充为0
            m = eval(f"df[col].groupby(cate_df[col]).{method_list[1]}()")
            num = df[col].groupby(cate_df[col]).count()
            m[num<thres] = 0
            mvalues = cate_df[col].replace(m.index,m)
            # 填充空值
            cond = df[col].isnull()
            df.loc[cond,col] = mvalues[cond]
    
    elif method in ["ffill","bfill"]:
        df = eval(f"df.{method}()")
    
    # 调整数据方向
    if axis == 1:
        df = df.T
    
    return df


# ================================================================================
# 标准化
# df            DataFrame
# method        标准化方式，["z-score","min-max","expanding-quantile","rolling-quantile","rank"]
# axis          计算方向，0为对列处理，1为对行处理
# rolling_days  滚动分位数的天数（第四种方法适用）
# rank_method   排序方法（第五种方法适用）
# ================================================================================
def standardize(df,method,axis=0,rolling_days=250,rank_method="average",meanExpo=None):

    # 初始化
    df = df.copy()
    
    # 设定数据处理方向
    if axis == 1:
        df = df.T
    
    if method == "z-score":
        df = (df-meanExpo)/df.std()
        
    elif method == "min-max":
        df = (df-df.min())/(df.max()-df.min())
    
    elif method == "expanding-quantile":
        rank_apply = lambda x: np.searchsorted(x,x[-1],sorter=np.argsort(x))/(len(x)-1)
        df = df.expanding().apply(rank_apply)
    
    elif method == "rolling-quantile":
        rank_apply = lambda x: np.searchsorted(x,x[-1],sorter=np.argsort(x))/(len(x)-1)
        df = df.rolling(rolling_days).apply(rank_apply)
        
    elif method == "rank":
        df = df.rank(method=rank_method)
    
    # 调整数据方向
    if axis == 1:
        df = df.T
    
    return df


# ================================================================================
# 按照wind的方式来执行利润表或现金流量表科目的转换
# ================================================================================
def transform_ttm_in_wind_way(df_factor):
    
    # 季频日期序列
    quaterly_dates = df_factor.index.to_list()
    
    # 目标截面日期区间（为了给同比计算保留裕量，这里从04年开始）
    panel_dates = df_factor.loc['2001':, :].index
    
    # 初始化返回值
    df_ret = pd.DataFrame(index=panel_dates, columns=df_factor.columns, dtype=float)
    
    # 遍历每一个截面
    for date in panel_dates:
        
        # 当前日期在季报序列中的位置
        q_loc = quaterly_dates.index(date)
        
        # 获取当前季报编号，并执行不同操作
        q_index = int(date.month/3) % 4
        
        # 如果是年报值，直接设置
        if q_index == 0:
            df_ret.loc[date, :] = df_factor.loc[date,:]
        
        # 如果是季报值，当前数据 + 去年年报 - 去年同期
        else:
            data_ttm = df_factor.iloc[q_loc-q_index,:] + \
                    df_factor.iloc[q_loc,:] - df_factor.iloc[q_loc-4,:]
                    
            # 计算出有空值则直接用去年年报再填充一遍
            data_ttm = data_ttm.fillna(df_factor.iloc[q_loc-q_index,:])
            
            # 保存结果
            df_ret.loc[date, :] = data_ttm
    
    return df_ret


# ================================================================================
# 回归法计算年复合增长率
# ================================================================================
def cal_cagr(df,n,negaopt="nan"):
    
    # 提取年报数据
    df_yearly = df.resample("Y").last()
    
    # 自变量为自然数序列
    input_seq = np.repeat(np.expand_dims(range(df_yearly.shape[0]),1),df_yearly.shape[1],1)
    input_seq = pd.DataFrame(input_seq,index=df_yearly.index,columns=df_yearly.columns)
    
    # 滚动计算回归系数
    _,coef,_ = rolling_weighted_regression(df_yearly,input_seq,n,pbar=False)
    
    # 滚动计算平均值
    denom = df_yearly.rolling(n).mean()
    
    # 分母为负时的调整
    if negaopt == "nan":
        denom[denom<0] = np.nan
    elif negaopt == "abs":
        denom = denom.abs()
    
    # 计算增长率
    res = coef/denom
    
    return res


# 生成2d-array的滚动窗口
def gen_rolling_windows(arr,window):
    rw = np.lib.stride_tricks.as_strided( \
        x=arr,shape=(arr.shape[0]-window+1,window,arr.shape[1]),
        strides=(arr.strides[0],arr.strides[0],arr.strides[1]))
    return rw


# 计算线性回归的系数和残差序列
def cal_resid(y,x,w):
    beta_est = np.linalg.inv(x.T @ w @ x) @ x.T @ w @ y
    resid = y - x @ beta_est
    return beta_est,resid


# nd-array转换为指定行列索引的dataframe
def array2dataframe(arr,df):
    start_ind = df.shape[0]-arr.shape[0]
    new_df = pd.DataFrame(arr,index=df.index[start_ind:],columns=df.columns)
    new_df = new_df.reindex(df.index)
    return new_df


# 对发布日序列进行提取     
def get_pct(cur_list):
          
    # 去除空值
    cur_list = cur_list.dropna()
    
    # 年报和次年一季报同时发布的场景，只保留一季报
    cur_list.drop_duplicates(keep='last', inplace=True)
        
    # 返回series，用于拼接成为矩阵
    output = pd.Series(cur_list.index, index=cur_list, name=cur_list.name)
            
    return output

# 填充季度索引
# def fill_Q(quarterLoc,df1,df2):
#     for i,istock in enumerate(quarterLoc.columns):
#         progress_bar(i+1,len(quarterLoc.columns))
#         a = pd.Series(df1.index).apply(lambda x: fill_quarter(x,quarterLoc.loc[:,istock],df2,istock))
#         a.index = df1.index    
#         df1.loc[:,istock] = a
#     return df1
# def fill_Q(quarterLoc,df1):
#     for i,istock in enumerate(quarterLoc.columns):
#         progress_bar(i+1,len(quarterLoc.columns))
#         a = pd.Series(df1.index).apply(lambda x: fill_quarter(x,quarterLoc.loc[:,istock]))
#         a.index = df1.index    
#         df1.loc[:,istock] = a
#     return df1

# # 填充季度索引子函数
# # def fill_quarter(x,df1,df2,istock):
# def fill_quarter(x,df1):   
#     #print(x)
#     data = pd.to_datetime(df1).tolist()
#     if x in data:
#         #print(x)
#         index = data.index(x)
#         #columns = istock
#         #return df2.loc[index,columns].values[0]
#         return index
#     else:
#         return np.nan



