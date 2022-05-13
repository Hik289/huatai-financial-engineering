import os 
import pandas as pd
import numpy as np


# -----------------------------------------------------------------------------
# 产业资本指标计算
# -----------------------------------------------------------------------------
class indus_cap():

    # -------------------------------------------------------------------------
    # 实例化对象，主要用于加载全局数据，避免每次重复加载
    # -------------------------------------------------------------------------
    def __init__(self):
        
        # 获取上级文件路径
        file_path = os.path.abspath(os.path.dirname(os.getcwd()))
        
        # 读取A股均价
        self.avg_price = pd.read_pickle(file_path + '/flow/data/daily/stock/S_DQ_AVGPRICE')
        
        # 读取A股收盘价
        self.stock_close = pd.read_pickle(file_path + '/flow/data/daily/stock/S_DQ_ADJCLOSE') # 复权
        self.stock_ori_close = pd.read_pickle(file_path + '/flow/data/daily/stock/S_DQ_CLOSE') # 不复权
        
        # 下载指数
        self.index_close = pd.read_pickle(file_path + '/flow/data/daily/index/index_close')        
        
        # 读取A股收盘价
        self.stock_close = pd.read_pickle(file_path + '/flow/data/daily/stock/S_DQ_ADJCLOSE') 

        # 读取A股成交额
        self.stock_amount = pd.read_pickle(file_path + '/flow/data/daily/stock/S_DQ_AMOUNT').loc['2005-01-01':,:] * 1000
        
        # 读取A股流通市值
        self.float_size = pd.read_pickle(file_path + '/flow/data/daily/stock/S_DQ_MV').loc['2005-01-01':,:] * 10000

        # 所有日频交易日期序列
        self.daily_dates = pd.Series(self.stock_close.index, index=self.stock_close.index)
    
        # 是否出现过借壳上市
        self.judge = (pd.read_csv(file_path + '/flow/data/借壳上市.csv', index_col=0, encoding='gbk') == '否').iloc[0,:]
    
        # 数据文件和简称对应关系
        self.name_dict = {"定向增发":"AShareSEO",
                          "限售股解禁":"AShareFreeFloatCalendar",
                          "股票回购":"AshareStockRepo",
                          "内部增减持":"AShareInsiderTrade",
                          "主要股东增减持":"AShareMjrHolderTrade",
                          "拟增减持":"ASarePlanTrade"}
            
        # 读取数据
        self.AShareSEO = pd.read_pickle(file_path + "/flow/data/money/AShareSEO") # 定向增发
        self.AShareFreeFloatCalendar = pd.read_pickle(file_path + "/flow/data/money/AShareFreeFloatCalendar") # 限售股解禁
        self.AshareStockRepo = pd.read_pickle(file_path + "/flow/data/money/AshareStockRepo") # 股票回购
        self.AShareInsiderTrade = pd.read_pickle(file_path + "/flow/data/money/AShareInsiderTrade") # 内部增减持  
        self.AShareMjrHolderTrade = pd.read_pickle(file_path + "/flow/data/money/AShareMjrHolderTrade") # 主要股东增减持
        self.ASarePlanTrade = pd.read_pickle(file_path + "/flow/data/money/ASarePlanTrade") # 拟增减持
              
        
    # ================================================================================
    # 数据读取
    # ================================================================================
    def data_preprocess(self, data_name, paras):
                
        # 定向增发
        if data_name == "AShareSEO":
                            
            # 相同时间数据去重
            cur_data = self.AShareSEO.sort_values('OPDATE')
            cur_data = cur_data.drop_duplicates(subset=['EVENT_ID'], keep='last') 
            
            # 预案公告日	S_FELLOW_PREPLANDATE
            # 股东大会公告日	S_FELLOW_SMTGANNCEDATE
            # 发审委/上市委通过公告日	S_FELLOW_PASSDATE
            # 证监会通过公告日	S_FELLOW_APPROVEDDATE
            # 增发公告日	S_FELLOW_OFFERINGDATE		
            
            para_dict = {'preplan':'S_FELLOW_PREPLANDATE',    # 预案公告日
                         'pass':'S_FELLOW_PASSDATE',          # 审核通过日
                         'offering':'S_FELLOW_OFFERINGDATE'}  # 增发公告日
            
            # 提取数据
            data = cur_data.loc[:,['S_INFO_WINDCODE', para_dict[paras], 
                             'S_FELLOW_PRICE','S_FELLOW_AMOUNT','EXP_COLLECTION']]
            
            # 募股数调整，计算实际募集金额
            data["S_FELLOW_AMOUNT"] = data['S_FELLOW_AMOUNT'] * 10000
            data["AMOUNT"] = data['S_FELLOW_PRICE'] * data['S_FELLOW_AMOUNT']
            
            # 列名替换
            data.columns = ['股票代码', '统计日期', '募集价格', '募集股数', '预期募集资金', '实际募集资金']
            data = data[~data.loc[:,'统计日期'].isnull()]
            data = data[~(data.loc[:,'股票代码'] == 'T00018.SH')]
            data = data.reset_index(drop=True)
            
            # 日期形式转换，只保留2005年以后的数据
            data['统计日期'] = [pd.to_datetime(str(int(i))) for i in data['统计日期']]
            data = data[data['统计日期'] < self.stock_ori_close.index[-1]]
            
            # 股票和行业价格数据进行填充
            stock_close = self.stock_ori_close.resample('D').ffill()
            
            # 有实际募集资金数据时，优先用实际募集资金（发行价格*发行数量）
            # 没有实际募集资金数据采用预期募集资金数据（EXP_COLLECTION）
            # 如果两者都没有，采用目标募集股份数*当日价格的80%进行估算    
            for index in data.index:
            
                # 记录预估发行价格
                data.loc[index, '预估发行价格'] = \
                stock_close.loc[data.loc[index, '统计日期'], data.loc[index, '股票代码']] * 0.8
                                               
            # 预估募集金额
            data['预估募集金额'] = data['预估发行价格'] * data['募集股数']
            
            # 只有在增发公告日会公布详细的募集金额，其他几个时间都采用预估的募集数据
            if para_dict[paras] != 'S_FELLOW_OFFERINGDATE':
                data.loc[:, '实际募集资金'] == 0
            
            # 预估募集资金汇总
            data['汇总金额'] = data['实际募集资金']
            data.loc[data['汇总金额'].isnull(),'汇总金额'] = data.loc[data['汇总金额'].isnull(),'预期募集资金'] 
            data.loc[data['汇总金额'].isnull(),'汇总金额'] = data.loc[data['汇总金额'].isnull(),'预估募集金额'] 
            
            # 结果汇总
            output = data.loc[:, ['股票代码', '统计日期', '汇总金额']]
            output.columns = ['股票代码', '统计日期', '资金数据']
                
            
        # 限售股解禁    
        elif data_name == "AShareFreeFloatCalendar":
            
            # 上市股份数量（万股）	S_SHARE_LST
            # 未上市股份数量（万股）	S_SHARE_NONLST
            
            # 提取数据
            data = self.AShareFreeFloatCalendar.loc[:,['S_INFO_WINDCODE',
                   'S_INFO_LISTDATE', 'ANN_DT', 'S_SHARE_LST', 'S_SHARE_NONLST']]
            
            # 列名替换
            data.columns = ['股票代码', '限售股上市日期', '公告日期', '上市股份数量', '未上市股份数量']
            data = data[~(data.loc[:,'股票代码'] == 'T00018.SH')]
            data = data.reset_index(drop=True)
            
            # 从万股转换成股
            data.loc[:,['上市股份数量', '未上市股份数量']] = data.loc[:,['上市股份数量', '未上市股份数量']] * 10000
            
            # 日期形式转换
            data['公告日期'] = pd.to_datetime(data['公告日期'].astype(str))
            data['限售股上市日期'] = [np.nan if pd.isnull(i) else pd.to_datetime(str(int(i))) for i in data['限售股上市日期']]  

            # 按照时间差距
            data['时间差距'] = data['限售股上市日期'] - data['公告日期']
            data['修正日期'] = data['限售股上市日期'] - pd.Timedelta(days=20)
            data.loc[data['时间差距'] > '20 days', '公告日期'] = data['修正日期'] 
            
            # 股票和行业价格数据进行填充
            stock_close = self.stock_ori_close.resample('D').ffill()
                    
            # 按照最近股价统计
            for index in data.index:
                
                if data.loc[index, '公告日期'] <= self.stock_ori_close.index[-1]:
                    
                    # 记录预估解禁价格
                    data.loc[index, '预估解禁价格'] = \
                    stock_close.loc[data.loc[index, '公告日期'], data.loc[index, '股票代码']]
                               
            # 预估募集金额 
            data['预估解禁金额'] = data['预估解禁价格'] * data['上市股份数量']
            
            # 按照公告日期提取数据
            if paras == 'anndt':             
                
                # 结果汇总
                output = data.loc[:, ['股票代码', '公告日期', '预估解禁金额']]
                output.columns = ['股票代码', '统计日期', '资金数据']
                
            if paras == 'listdt':   
                
                # 结果汇总
                data = data[~(data['时间差距'] < '0 days')]
                output = data.loc[:, ['股票代码', '公告日期',  '限售股上市日期', '上市股份数量']]  
                output.columns = ['股票代码', '统计日期',  '限售股上市日期', '上市股份数量']
          
            
        # 股票回购
        elif data_name == "AshareStockRepo":
             
            # 每个回购时间持续时间过长，不适合进行长时间统计
            # 着重统计三个时间点：董事会预案、股东大会通过、中间实施回购的日期（可能有多个）
        
            # 提取数据
            data = self.AshareStockRepo.loc[:,['S_INFO_WINDCODE', 'ANN_DT', 
                                      'AMT', 'TOTAL_SHARE_RATIO', 'STATUS']]
            
            # 列名替换
            data.columns = ['股票代码', '公告日期', '回购金额',
                            '回购股数占总股本比例', '进度类型代码']
            data = data.reset_index(drop=True)
            
            # # 提取类型
            # 324003002	董事会预案
            # 324003004	股东大会通过
            # 324004001	回购实施                    
            para_dict = {'preplan':324003002,   # 324003002	董事会预案
                         'pass':324003004,      # 324003004	股东大会通过
                         'conduct':324004001}   # 324004001	回购实施
            
            data = data[data['进度类型代码'] == para_dict[paras]]
        
            # 日期类型转换
            data['公告日期'] = pd.to_datetime(data['公告日期'].astype(str))
            data = data[data['公告日期'] < self.stock_ori_close.index[-1]]
                               
            # 结果汇总
            output = data.loc[:, ['股票代码', '公告日期', '回购金额']]
            output.columns = ['股票代码', '统计日期', '资金数据']

  
        # 主要股东增减持
        elif data_name == "MjrHolderTrade":
                                
            # 提取数据
            data_mjr = self.AShareMjrHolderTrade.loc[:,
                          ['S_INFO_WINDCODE','ANN_DT', 'TRANSACT_QUANTITY',
                           'TRANSACT_TYPE', 'AVG_PRICE', 'IS_REANNOUNCED']]
            
            # 列名替换
            data_mjr.columns = ['股票代码', '公告日期', '变动数量', 
                                '买卖方向', '均价', '是否重复']
            
            # 剔除重复数据
            data_mjr_diff = data_mjr[~(data_mjr['是否重复']==1)] 
            
            # 替换形式
            data_mjr_diff = data_mjr_diff.reset_index(drop=True)    
            data_mjr_diff['公告日期'] = pd.to_datetime(data_mjr_diff['公告日期'])
            
            # 股票价格数据进行填充
            stock_close = self.stock_ori_close.resample('D').ffill()
                                           
            # 记录预估发行价格          
            for index in data_mjr_diff.index:
                
                if np.isnan(data_mjr_diff.loc[index, '均价']):
                    
                    data_mjr_diff.loc[index, '均价'] = stock_close.loc[
                            data_mjr_diff.loc[index, '公告日期'],
                            data_mjr_diff.loc[index, '股票代码']]    
        
            # 提取数据
            data_inside = self.AShareInsiderTrade.loc[:,['S_INFO_WINDCODE', 
                      'ACTUAL_ANN_DT', 'CHANGE_VOLUME',  'TRADE_AVG_PRICE']]
            data_inside = data_inside.reset_index(drop=True)  
            
            # 列名替换
            data_inside.columns = ['股票代码', '公告日期', '变动数量', '均价']
                        
            # 日期类型转换
            data_inside['公告日期'] = pd.to_datetime(data_inside['公告日期'])
            
            # 计算变动金额
            data_inside.loc[data_inside['变动数量']>0, '买卖方向'] = '增持'
            data_inside.loc[data_inside['变动数量']<0, '买卖方向'] = '减持'
            data_inside['变动数量'] = data_inside['变动数量'].abs()
            
            # 数据合并
            merge_data = pd.concat([data_mjr_diff, data_inside], axis=0)
            merge_data = merge_data.reset_index(drop=True)    
            
            # 预估募集金额
            merge_data['变动金额'] = merge_data['变动数量'] * merge_data['均价']
            
            # 相同数据合并
            map_dict = {'over':'增持', 'under':'减持'}
            merge_data = merge_data[merge_data['买卖方向'] == map_dict[paras]]
            merge_data_group = merge_data.groupby(['股票代码', '公告日期']).sum()['变动金额'].reset_index()    
               
            # 结果汇总
            output = merge_data_group.loc[:, ['股票代码', '公告日期', '变动金额']]
            output.columns = ['股票代码', '统计日期', '资金数据']
        
                          
        # 主要股东增减持
        elif data_name == "AShareMjrHolderTrade":
            
            # 提取数据
            data = self.AShareMjrHolderTrade.loc[:,['S_INFO_WINDCODE', 
                        'ANN_DT', 'TRANSACT_QUANTITY',  'TRANSACT_TYPE']]
            
            # 列名替换
            data.columns = ['股票代码', '公告日期', '变动数量', '买卖方向']
            data['公告日期'] = pd.to_datetime(data['公告日期'].astype(str))
            data = data.reset_index(drop=True)
            
            # 提取类型
            data = data[data['买卖方向'] == paras]
            
            # 相同日期数据合并
            data = data.groupby(['股票代码', '公告日期']).sum()['变动数量'].reset_index()    
            data = data[data['公告日期'] < self.stock_ori_close.index[-1]]
                                              
            # 股票和行业价格数据进行填充
            stock_close = self.stock_ori_close.resample('D').ffill()
                        
            for index in data.index:
            
                # 记录预估发行价格
                data.loc[index, '估计变动价格'] = \
                stock_close.loc[data.loc[index, '公告日期'], data.loc[index, '股票代码']]
                                               
            # 预估募集金额
            data['估计变动金额'] = data['估计变动价格'] * data['变动数量']
                                                
            # 日期类型转换
            data['公告日期'] = pd.to_datetime(data['公告日期'].astype(str))
            
            # 结果汇总
            output = data.loc[:, ['股票代码', '公告日期', '估计变动金额']]
            output.columns = ['股票代码', '统计日期', '资金数据']
            
            
        # 内部增减持
        elif data_name == "AShareInsiderTrade":
            
            # 提取数据
            data = self.AShareInsiderTrade.loc[:,['S_INFO_WINDCODE', 
              'ACTUAL_ANN_DT', 'CHANGE_VOLUME',  'TRADE_AVG_PRICE']]
            
            # 列名替换
            data.columns = ['股票代码', '公告日期', '变动数量', '成交均价']
                        
            # 日期类型转换
            data['公告日期'] = pd.to_datetime(data['公告日期'].astype(str))
            
            # 计算变动金额
            data['变动金额'] = data['变动数量'] * data['成交均价'] 
            data = data.reset_index(drop=True)
            
            # 提取类型
            if paras == '增持':
                output_data = data[data['变动金额'] > 0]
            elif paras == '减持':
                output_data = data[data['变动金额'] < 0]
                output_data.loc[:, '变动金额'] = - output_data.loc[:, '变动金额'].values
                
            # 结果汇总
            output = output_data.loc[:, ['股票代码', '公告日期', '变动金额']]
            output.columns = ['股票代码', '统计日期', '资金数据']  
                        
            
        # 拟增减持
        elif data_name == "ASarePlanTrade":
                    
            # 提取数据
            data = self.ASarePlanTrade.loc[:,['S_INFO_WINDCODE', 'ANN_DT',
                    'ANN_DT_NEW', 'HOLDER_NAME', 'PROGRAMME_PROGRESS', 'TRANSACT_TYPE',
                    'CHANGE_START_DATE',  'CHANGE_END_DATE', 'PLAN_TRANSACT_MAX_NUM']]
                 
            # 列名替换
            data.columns = ['股票代码', '首次披露公告日', '统计日期', '持有方名称', '方案进度',
                            '变动方向', '变动起始日期', '变动截止日期', '拟变动数量上限']
            data = data.reset_index(drop=True)
            
            # 剔除空值
            data = data[data['变动起始日期'].notnull()]
            data = data[data['变动截止日期'].notnull()]
            data = data[data['拟变动数量上限'].notnull()]
                    
            # 日期形式转换
            data['变动起始日期'] = [pd.to_datetime(str(int(i))) for i in data['变动起始日期']]
            data['变动截止日期'] = [pd.to_datetime(str(int(i))) for i in data['变动截止日期']]
            data['首次披露公告日'] = [pd.to_datetime(str(int(i))) for i in data['首次披露公告日']]
            data['统计日期'] = [pd.to_datetime(str((i))) for i in data['统计日期']]
            
            # 当最新状态为完成或是失败时，截止日期替换为最新公告日
            data.loc[(data['方案进度']==324004000) | (data['方案进度']==324005000), '变动截止日期'] = data['统计日期']    
            
            # 计算减持数据
            map_dict = {'over':'增持', 'under':'减持'}
            data = data[data['变动方向'] == map_dict[paras]]
            
            output = data
            
        return output
            
    
    # -------------------------------------------------------------------------
    # 按周度或月度聚合数据
    #
    # origin_data:原始数据
    # freq:频率
    # method: sum/last
    # day:平移天数（-1表示未来一周或一月，0表示过去）
    # -------------------------------------------------------------------------   
    def data_resample(self, origin_data, freq, method, day):
        
        # 取不同频率数据
        data_freq = origin_data.resample(freq).sum().shift(day)

        if method == 'sum':
                
            # 取不同频率数据，求和
            data_freq = origin_data.resample(freq).sum().shift(day)
            
        elif method == 'last':
                                
            # 取不同频率数据
            origin_data_change = origin_data.loc[origin_data.sum(axis=1)!=0, :]
            data_freq = origin_data_change.resample(freq).last().shift(day)
            
            
        # 日期调整，汇总到最近的交易日上
        change_dates = self.daily_dates.resample(freq).last()
            
        # 舍弃数据
        data_freq = data_freq.loc[data_freq.index <= change_dates.index[-1],:]
        
        # 替换名称
        data_freq.index = pd.to_datetime(change_dates[data_freq.index].values)
        
        # 剔除空值数据
        data_freq = data_freq.loc[data_freq.index.notnull(), :]
        
        return data_freq
    

    
    # -------------------------------------------------------------------------
    # 方案一，汇总近期资金流向事件
    # 方案二，汇总未来一段时间资金流向
    # -------------------------------------------------------------------------   
    def gen_factor(self, data_name, data, period_type, data_type, freq):
        
        # 'recent':不平移, 'next':平移一期
        day_shift = 1 if period_type == 'next' else 0   
        
        # 计算行业资金流数据
        if (data_name == 'AShareFreeFloatCalendar') & (period_type == 'next'):
            
            indus_money_freq = self.gen_future_calendar(data, freq)
            
        elif (data_name == 'ASarePlanTrade') & (period_type == 'next'):
            
            indus_money_freq = self.gen_future_planhold(data, freq)
            
        else:
                
            # 分组聚合
            indus_money = data.groupby(['股票代码', '统计日期']).sum().\
                reset_index().pivot('统计日期','股票代码','资金数据')
                
            # 列名和日期重置
            indus_money = indus_money.reindex(columns=self.stock_amount.columns).\
                resample('D').last()
            
            # 计算资金流指标
            indus_money_freq = self.data_resample(indus_money, freq, 'sum', day_shift)
                    
                    
        # 读取流通市值，计算资金流数据和行业流通市值之比
        if data_type == 'float':

            indus_float = self.float_size
            indus_float_freq = self.data_resample(indus_float, freq, 'sum', 0)
            indus_money_change = indus_money_freq / indus_float_freq
            
        # 读取成交额，计算资金流数据和行业成交额之比            
        if data_type == 'amount':

            indus_amount = self.stock_amount
            indus_amount_freq = self.data_resample(indus_amount, freq, 'sum', 0)
            indus_money_change = indus_money_freq / indus_amount_freq            
            
        # 原始财务数据
        elif data_type == 'orig':
                             
            indus_money_change = indus_money_freq
                                    
        return indus_money_change
     

    # -------------------------------------------------------------------------
    # 预期金额
    # -------------------------------------------------------------------------   
    def gen_future_calendar(self, data, freq):
            
        # 周度为5个交易日，月度为20个交易日
        next_days = 5 if freq == 'W' else 20
        
        # 日期频率
        resample_date = self.daily_dates.resample(freq).last().loc['2005-01-01':].dropna()
       
        # 记录行业数据
        indus_data = pd.DataFrame(np.nan, index=resample_date.values, \
                                  columns=self.stock_amount.columns)    
        for index in resample_date.index:
            
            # 已经公告的日期
            select_data = data[data['统计日期']<=index]
            
            # 日期序列
            day_position = self.daily_dates.index.tolist().index(resample_date[index])
            
            # 超出数据范围按日历日范围选取数据
            if (day_position+next_days) >= len(self.daily_dates):                
                reselect_data = select_data[(select_data['限售股上市日期'] > index)& (
                    select_data['限售股上市日期']<=(index + pd.Timedelta(days=int(next_days/5+7))))]
                
            else:
                
                # 提取下一期数据
                reselect_data = select_data[(select_data['限售股上市日期'] > index) & (
                    select_data['限售股上市日期']<=self.daily_dates.iloc[day_position+next_days])]
                
            if len(reselect_data) == 0:
                indus_data.loc[resample_date[index], :] = np.nan
                continue                
            else:
                
                # 相同日期合并
                stock_volumes  = reselect_data.groupby(['股票代码', '限售股上市日期']).sum()['上市股份数量']

                tmp_data = stock_volumes.reset_index().set_index('股票代码')
                
                # 记录收盘价数据
                tmp_data.loc[:, '收盘价'] = self.stock_ori_close.loc[
                    resample_date[index], tmp_data.index]
                            
                # 计算预估解禁金额
                tmp_data.loc[:, '资金数据'] = tmp_data.loc[:, '收盘价'] * tmp_data.loc[:, '上市股份数量']
                                
                # 结果存储
                tmp_data = tmp_data.reset_index().pivot('限售股上市日期','股票代码','资金数据').\
                    resample(freq).sum()
                
                for i in tmp_data.index:
                    if i in resample_date.index:
                        
                        indus_data.loc[resample_date[i],tmp_data.columns] = tmp_data.values[0]
                
        return indus_data
            
            
    
    
    # -------------------------------------------------------------------------
    # 预期金额
    # -------------------------------------------------------------------------   
    def gen_future_planhold(self, data, freq):
                        
        # 周度为5个交易日，月度为20个交易日
        next_days = 5 if freq == 'W' else 20
        
        # 日期频率
        resample_date = self.daily_dates.resample(freq).last().loc['2005-01-01':].dropna()
        # 记录行业数据
        indus_data = pd.DataFrame(np.nan, index=resample_date.values, \
                                  columns=self.stock_amount.columns)  
        
        for index in resample_date.index:
            
            # 已经公告的日期
            select_data = data[data['统计日期']<=index]
            
            # 日期序列
            day_position = self.daily_dates.index.tolist().index(resample_date[index])
            
            # 提取下一期数据，超出数据范围按日历日范围选取数据
            if (day_position+next_days) >= len(self.daily_dates):             
                cur_date = index + pd.Timedelta(days=int(next_days/5+7))
                
            else:
                cur_date = self.daily_dates.iloc[day_position+next_days]
            
            # 筛选当前处于执行中的计划
            reselect_data = select_data[(select_data['变动起始日期'] < index) & (
                                         select_data['变动截止日期'] >= cur_date)]
            
            # 按最新公告日排序
            reselect_data = reselect_data.sort_values(by='统计日期', ascending=False)
            
            # 对筛选出的增减持计划进行去重
            keep_data = reselect_data.groupby(['股票代码', '首次披露公告日',
                               '持有方名称']).last().reset_index(drop=True)
                            
            if len(keep_data) > 0:
                    
                # 相同股票合并
                stock_volumes = reselect_data.groupby(['股票代码','统计日期']).sum()['拟变动数量上限']

                tmp_data = stock_volumes.reset_index().set_index('股票代码')
                
                # 记录收盘价数据
                tmp_data.loc[:, '收盘价'] = self.stock_ori_close.loc[
                    resample_date[index], tmp_data.index]
                            
                # 计算预估解禁金额
                tmp_data.loc[:, '资金数据'] = tmp_data.loc[:, '收盘价'] * tmp_data.loc[:, '拟变动数量上限']
                                
                # 结果存储
                tmp_data = tmp_data.reset_index().pivot('统计日期','股票代码','资金数据').\
                    resample(freq).sum()
                
                for i in tmp_data.index:
                    if i in resample_date.index:
                        
                        indus_data.loc[resample_date[i],tmp_data.columns] = tmp_data.values[0]
                        
        return indus_data
    
        
if __name__ == "__main__":
    
# =============================================================================
#   指标计算
# =============================================================================

    # 模型初始化
    model = indus_cap()
      
    # 生成样本空间
    settings_list = pd.DataFrame(        
        [   
            # 定向增发 AShareSEO 
            ['AShareSEO',  'preplan',  'recent'],   # 预案公告日
            ['AShareSEO',  'pass',  'recent'],      # 审核通过日
            ['AShareSEO',  'offering',  'recent'],  # 增发公告日
            
            # 限售股解禁 AShareFreeFloatCalendar
            ['AShareFreeFloatCalendar',  'anndt',  'recent'], # 近期解禁股
            ['AShareFreeFloatCalendar',  'listdt',  'next'],  # 未来一期解禁股
            
            # 股票回购 AshareStockRepo
            ['AshareStockRepo',  'preplan',  'recent'], # 董事会预案
            ['AshareStockRepo',  'pass',  'recent'],    # 股东大会通过
            ['AshareStockRepo',  'conduct',  'recent'], # 回购实施
            
            # 内部增减持 + 主要股东增减持
            ['MjrHolderTrade',  'over',  'recent'],  # 近期增持
            ['MjrHolderTrade',  'under',  'recent'], # 近期减持
            
            # 拟增减持 ASarePlanTrade
            ['ASarePlanTrade',  'over',  'next'],    # 未来一期增持
            ['ASarePlanTrade',  'under',  'next']    # 未来一期减持        
        ],
        
    columns=['数据名称', '日期参数','时间设定'])
       
    
    for index in settings_list.index:
        
        # 基础数据设定
        data_name = settings_list.loc[index, '数据名称']
        paras = settings_list.loc[index, '日期参数']
        period_type = settings_list.loc[index, '时间设定']
            
        # 原始数据处理
        data_pos = model.data_preprocess(data_name, paras)
              
        
        # 计算指标
        for data_type in ['orig', 'float', 'amount']:
            for freq in ['W', 'M']:
                
                # 计算数据名称，数据对应参数，上期还是下期设定，归一化形式，频率
                print(data_name, paras, period_type, data_type, freq)
                
                # 数据计算和存储
                factor = model.gen_factor(data_name, data_pos, period_type, data_type, freq)
                
                # 产业资本数据经常为空值，0值直接置为空
                factor[factor==0] = np.nan
                
                # 结果存储
                factor.to_pickle(file_path+'/flow/results/{}_{}_{}_{}_{}'.format(
                    data_name, paras, period_type, data_type, freq))
                     
                
# =============================================================================
#   数据整理
# =============================================================================
    
    # # settings = ['AshareStockRepo',  'preplan',  'recent']    
    # # settings = ['AShareFreeFloatCalendar',  'listdt',  'next']
    # # settings = ['AShareSEO',  'preplan',  'recent']
    # # settings = ['MjrHolderTrade',  'under',  'recent']    
    # # settings = ['AShareFreeFloatCalendar',  'listdt',  'next']    
    # settings = ['ASarePlanTrade', 'over',  'next']    

    # # AShareFreeFloatCalendar_listdt_next_orig_M
    
    # # 基础数据设定
    # data_name = settings[0]
    # paras = settings[1]
    # period_type = settings[2]
        
    # # 原始数据处理
    # data_pos = model.data_preprocess(data_name, paras)

    # # 股票
    # data_pos = data_pos.loc[data_pos['统计日期']>pd.to_datetime('2005-01-01'),:]    
           
    # # 得到行业标签
    # data = model.get_indus_data(data_pos)    
    
    # # 计算因子值
    # factor = model.gen_factor(data_name, data, 'next', 'float', 'M')
    
    
    
        
