% -------------------------------------------------------------------------
% 批量测试不同胜率的模拟策略在增强场景下的表现
% -------------------------------------------------------------------------
clear; clc; close all;
addpath('utils');   
dbstop if error

% -------------------------------------------------------------------------
% 设置参数(参数在ParamSet文件中都有默认设置，若要覆盖则在本脚本中重新赋值)
% -------------------------------------------------------------------------
% 读取默认参数
param = ParamSet();

% 股指选择：沪深300(HS300)、中证500(ZZ500)
param.targetIndex = 'HS300';  

% 生成行业观点时是否采用指数内成分股合成下期收益，否则直接用一级行业
param.simuByStock = true;

% 是否目标指数内选股（1=是，0=否），否则在全A股内配置
param.selectInIndex = 1;   

% 收益调整系数：包含0、0.5、1、2四种场景
param.beta = 0;

% 模拟次数
param.simuNum = 100;

% 胜率列表
accuracyList = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1];

% -------------------------------------------------------------------------
% 数据导入
% -------------------------------------------------------------------------
% 获取股票基本信息、日频信息、月频信息
load('data/stock_data','basic_info','daily_info','monthly_info');

% 获取因子暴露以及协方差估计结果
load('result/factorExpo.mat');
factorCov = importdata('result/factorCovEigenAdj.mat');
specialCov = importdata('result/specialCovBiasAdj.mat');

% 获取基准基准指数信息
load('data/index_data.mat')
indexInfo = getfield(index_data,param.targetIndex);
indexClose = getfield(indexInfo.close,param.baseType)'; 

% 获取各行业指数每日收盘价
load('data/indus_data.mat');
indusClose = indus_data.indus_close;
clear indus_data

% 获取个股收益率预测(如果指数内选股则获取指数内预测结果，否则全市场预测)
if param.selectInIndex
    forecastReturn = importdata(sprintf('data/%s.mat',param.targetIndex));
else
    forecastReturn = importdata('data/market.mat');
end

% 计算基准组合权重矩阵
fullStockWeight = CalcFullStockWeight(indexInfo,basic_info);

% 每个截面上可交易的股票
validStockMatrix = CalcValidStockMatrix(basic_info,daily_info);

% 计算截面日期在日频日期中的索引
dailyDates = daily_info.dates;
dailyClose = daily_info.close_adj;
[~,param.month2day] = ismember(monthly_info.dates,dailyDates);

% -------------------------------------------------------------------------
% 获取基准组合
% -------------------------------------------------------------------------
% 行业中性组合
neutralPort = CalcNeutralWeight(param,fullStockWeight,validStockMatrix,...
                        forecastReturn,factorExpo,factorCov,specialCov);
[neutralNav,bktestDates] = CalcStrategyNav(param,neutralPort,dailyClose,dailyDates);

% 原指数净值
[~,bktestIndex] = ismember(bktestDates,dailyDates);
indexNav = indexClose(bktestIndex);
indexNav = indexNav ./ indexNav(1);

% -------------------------------------------------------------------------
% 蒙特卡洛模拟测试取平均来获得增强策略表现
% -------------------------------------------------------------------------
% 初始化结果
perf = {'年化收益','年化波动','夏普比率','最大回撤',...
        '相比中性年化超额','年化波动','信息比率','最大回撤',...
        '相比指数年化超额','年化波动','信息比率','最大回撤'};

% 遍历每个胜率水平
for iAccuracy = 1:length(accuracyList)
    
    % 获取当前胜率
    param.accuracy = accuracyList(iAccuracy);
    fprintf('当前胜率：%.0f%%\n',100*param.accuracy);
    
    % 存储每次模拟的业绩指标，依次为：绝对收益、相比于中性策略、相比于指数基准
    enhancePerf = nan(param.simuNum,12);

    % 遍历每次模拟
    for simu = 1:param.simuNum

        fprintf('    第%d次模拟\n',simu);

        % 生成行业预测观点
        if param.simuByStock
            indusView = CalcSimuViewByStock(param,dailyClose,fullStockWeight,factorExpo);
        else
            indusView = CalcSimulateIndusView(param,indusClose);
        end

        % 生成带行业观点的增强组合
        port = CalcEnhanceWeight(param,fullStockWeight,validStockMatrix,...
                    forecastReturn,factorExpo,indusView,factorCov,specialCov);

        % 存储策略净值
        [nav,~] = CalcStrategyNav(param,port,dailyClose,dailyDates);

        % 计算策略绝对收益表现
        perf1 = CalcStrategyPerf(nav);

        % 计算策略相比于中性组合相对收益表现
        perf2 = CalcStrategyPerf(ret2tick(tick2ret(nav)-tick2ret(neutralNav)));

        % 计算策略相比于指数基准相对收益表现
        perf3 = CalcStrategyPerf(ret2tick(tick2ret(nav)-tick2ret(indexNav)));

        % 存储结果
        enhancePerf(simu,:) = cell2mat([perf1(2,:),perf2(2,:),perf3(2,:)]);

    end
    
    % 将多次模拟的结果取平均
    perf = cat(1,perf,num2cell( mean(enhancePerf)));

end

% 添加行名
rows = arrayfun(@(x) sprintf('%.0f%%',100*x),accuracyList,'UniformOutput',false);
perf = [[{''};rows'],perf];



