% -------------------------------------------------------------------------
% 纯行业中性测试
% -------------------------------------------------------------------------
clear; clc; close all;
addpath('utils');    
dbstop if error

% -------------------------------------------------------------------------
% 设置参数(参数在ParamSet文件中都有默认设置，若要覆盖则在本脚本中重新赋值)
% -------------------------------------------------------------------------
% 读取参数
param = ParamSet();

% 股指选择：沪深300(HS300)、中证500(ZZ500)
param.targetIndex = 'HS300';   

% 是否目标指数内选股（1=是，0=否），否则在全A股内配置
param.selectInIndex = 1;   

% 风险厌恶系数，取值为零表示进行线性规划求解，否则二次规划求解（运行很慢）
param.lambda = 0; 

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
% 执行回测
% -------------------------------------------------------------------------
% 生成权重组合
port = (param,fullStockWeight,validStockMatrix,...
                         forecastReturn,factorExpo,factorCov,specialCov);

% 计算净值
[nav,dates] = CalcStrategyNav(param,port,dailyClose,dailyDates);

% 该模拟的收益表现
perf = CalcStrategyPerf(nav);

% 计算相比于指数基准的超额收益表现
[~,index] = ismember(dates,dailyDates);
indexNav = indexClose(index);
excess = ret2tick(tick2ret(nav)-tick2ret(indexNav));
excess_perf = CalcStrategyPerf(excess);

% -------------------------------------------------------------------------
% 作图
% -------------------------------------------------------------------------
figure;
hold on
plot(excess,'r');
dateStr = cellstr(datestr(dates,'yyyy-mm'));
step = ceil(length(dateStr)/12);
set(gca,'xtick',1:step:length(dateStr),'xTickLabel',dateStr(1:step:end),'xTickLabelRotation',50);
set(gcf,'color','w'); 
set(gcf,'position',[350,150,600,400])
set(gca,'FontSize',10,'FontWeight','bold','FontName','STFangSong','box','off');
xlim([1 length(dateStr)]);


