% -------------------------------------------------------------------------
% ����ҵ���Բ���
% -------------------------------------------------------------------------
clear; clc; close all;
addpath('utils');    
dbstop if error

% -------------------------------------------------------------------------
% ���ò���(������ParamSet�ļ��ж���Ĭ�����ã���Ҫ�������ڱ��ű������¸�ֵ)
% -------------------------------------------------------------------------
% ��ȡ����
param = ParamSet();

% ��ָѡ�񣺻���300(HS300)����֤500(ZZ500)
param.targetIndex = 'HS300';   

% �Ƿ�Ŀ��ָ����ѡ�ɣ�1=�ǣ�0=�񣩣�������ȫA��������
param.selectInIndex = 1;   

% �������ϵ����ȡֵΪ���ʾ�������Թ滮��⣬������ι滮��⣨���к�����
param.lambda = 0; 

% -------------------------------------------------------------------------
% ���ݵ���
% -------------------------------------------------------------------------
% ��ȡ��Ʊ������Ϣ����Ƶ��Ϣ����Ƶ��Ϣ
load('data/stock_data','basic_info','daily_info','monthly_info');

% ��ȡ���ӱ�¶�Լ�Э������ƽ��
load('result/factorExpo.mat');
factorCov = importdata('result/factorCovEigenAdj.mat');
specialCov = importdata('result/specialCovBiasAdj.mat');

% ��ȡ��׼��׼ָ����Ϣ
load('data/index_data.mat')
indexInfo = getfield(index_data,param.targetIndex);
indexClose = getfield(indexInfo.close,param.baseType)'; 

% ��ȡ����ҵָ��ÿ�����̼�
load('data/indus_data.mat');
indusClose = indus_data.indus_close;
clear indus_data

% ��ȡ����������Ԥ��(���ָ����ѡ�����ȡָ����Ԥ����������ȫ�г�Ԥ��)
if param.selectInIndex
    forecastReturn = importdata(sprintf('data/%s.mat',param.targetIndex));
else
    forecastReturn = importdata('data/market.mat');
end

% �����׼���Ȩ�ؾ���
fullStockWeight = CalcFullStockWeight(indexInfo,basic_info);

% ÿ�������Ͽɽ��׵Ĺ�Ʊ
validStockMatrix = CalcValidStockMatrix(basic_info,daily_info);

% ���������������Ƶ�����е�����
dailyDates = daily_info.dates;
dailyClose = daily_info.close_adj;
[~,param.month2day] = ismember(monthly_info.dates,dailyDates);

% -------------------------------------------------------------------------
% ִ�лز�
% -------------------------------------------------------------------------
% ����Ȩ�����
port = (param,fullStockWeight,validStockMatrix,...
                         forecastReturn,factorExpo,factorCov,specialCov);

% ���㾻ֵ
[nav,dates] = CalcStrategyNav(param,port,dailyClose,dailyDates);

% ��ģ����������
perf = CalcStrategyPerf(nav);

% ���������ָ����׼�ĳ����������
[~,index] = ismember(dates,dailyDates);
indexNav = indexClose(index);
excess = ret2tick(tick2ret(nav)-tick2ret(indexNav));
excess_perf = CalcStrategyPerf(excess);

% -------------------------------------------------------------------------
% ��ͼ
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


