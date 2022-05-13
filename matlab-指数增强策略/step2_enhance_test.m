% -------------------------------------------------------------------------
% 引入行业轮动观点进行指数增强测试
% -------------------------------------------------------------------------
clear; clc; close all;
addpath('utils');    
dbstop if error

% -------------------------------------------------------------------------
% 设置参数(参数在ParamSet文件中都有默认设置，若要覆盖则在本脚本中重新赋值)
% -------------------------------------------------------------------------
% 读取默认参数
param = ParamSet();

% 目标行业轮动策略观点文件
% param.file_name = 'data/全行业策略.xlsx';
% param.file_name = 'data/沪深300策略.xlsx';
param.file_name = 'data/中证500策略.xlsx';
% param.file_name = 'data/沪深300策略二.xlsx';

% 股指选择：沪深300(HS300)、中证500(ZZ500)
param.targetIndex = 'ZZ500'; 

% 是否目标指数内选股（1=是，0=否），否则在全A股内配置
param.selectInIndex = 1;

% 收益调整系数
param.beta = 2;

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

% 读取行业轮动策略观点
indusView = CalcRealIndusView(param,monthly_info);

% -------------------------------------------------------------------------
% 优化回测
% -------------------------------------------------------------------------
% 生成组合1：也即带行业观点的增强组合
enhancePort = CalcEnhanceWeight(param,fullStockWeight,validStockMatrix,...
    forecastReturn,factorExpo,indusView,factorCov,specialCov);
[enhanceNav,bktestDates] = CalcStrategyNav(param,enhancePort,dailyClose,dailyDates);

% 生成组合2：保持行业中性的选股组合
neutralPort = CalcNeutralWeight(param,fullStockWeight,validStockMatrix,...
    forecastReturn,factorExpo,factorCov,specialCov);
[neutralNav,~] = CalcStrategyNav(param,neutralPort,dailyClose,dailyDates);

% 生成组合3：也即原指数净值
[~,bktestIndex] = ismember(bktestDates,dailyDates);
indexNav = indexClose(bktestIndex);
indexNav = indexNav ./ indexNav(1);

% 计算超额净值
enhance2index = ret2tick(tick2ret(enhanceNav)-tick2ret(indexNav));
enhance2neutral = ret2tick(tick2ret(enhanceNav)-tick2ret(neutralNav));
neutral2index = ret2tick(tick2ret(neutralNav)-tick2ret(indexNav));

% 生成业绩指标
perf = CalcStrategyPerf([enhanceNav,neutralNav,indexNav,enhance2index,enhance2neutral,neutral2index]);
perf = [{'';'增强';'中性';'指数';'增强-指数';'增强-中性';'中性-指数'},perf];

% -------------------------------------------------------------------------
% 作图1：增强组合相比于指数基准的超额表现
% -------------------------------------------------------------------------
figure;
hold on
yyaxis left;
plot(enhanceNav,'r');
plot(indexNav,'b-');
yyaxis right;
plot(enhance2index,'c');
dateStr = cellstr(datestr(bktestDates,'yyyy-mm'));
step = ceil(length(dateStr)/8);
set(gca,'xtick',1:step:length(dateStr),'xTickLabel',dateStr(1:step:end),'xTickLabelRotation',50);
set(gcf,'color','w'); 
set(gcf,'position',[350,150,600,400])
legend({'增强','指数','超额'},'location','northwest');
set(gca,'FontSize',10,'FontWeight','bold','FontName','STFangSong','box','off');
xlim([1 length(dateStr)]);

% -------------------------------------------------------------------------
% 作图2：增强组合相比于中性组合的超额表现
% -------------------------------------------------------------------------
% 作图
figure;
hold on
yyaxis left;
plot(enhanceNav,'r');
plot(neutralNav,'b-');
yyaxis right;
plot(enhance2neutral,'c');
dateStr = cellstr(datestr(bktestDates,'yyyy-mm'));
step = ceil(length(dateStr)/8);
set(gca,'xtick',1:step:length(dateStr),'xTickLabel',dateStr(1:step:end),'xTickLabelRotation',50);
set(gcf,'color','w'); 
set(gcf,'position',[350,150,600,400])
legend({'增强','中性','超额'},'location','northwest');
set(gca,'FontSize',10,'FontWeight','bold','FontName','STFangSong','box','off');
xlim([1 length(dateStr)]);

% -------------------------------------------------------------------------
% 作图3：中性组合相比于指数基准的表现
% -------------------------------------------------------------------------
figure;
hold on
yyaxis left;
plot(neutralNav,'r');
plot(indexNav,'b-');
yyaxis right;
plot(neutral2index,'c');
dateStr = cellstr(datestr(bktestDates,'yyyy-mm'));
step = ceil(length(dateStr)/8);
set(gca,'xtick',1:step:length(dateStr),'xTickLabel',dateStr(1:step:end),'xTickLabelRotation',50);
set(gcf,'color','w'); 
set(gcf,'position',[350,150,600,400])
legend({'中性','指数','超额'},'location','northwest');
set(gca,'FontSize',10,'FontWeight','bold','FontName','STFangSong','box','off');
xlim([1 length(dateStr)]);
