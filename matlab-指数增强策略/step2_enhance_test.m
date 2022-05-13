% -------------------------------------------------------------------------
% ������ҵ�ֶ��۵����ָ����ǿ����
% -------------------------------------------------------------------------
clear; clc; close all;
addpath('utils');    
dbstop if error

% -------------------------------------------------------------------------
% ���ò���(������ParamSet�ļ��ж���Ĭ�����ã���Ҫ�������ڱ��ű������¸�ֵ)
% -------------------------------------------------------------------------
% ��ȡĬ�ϲ���
param = ParamSet();

% Ŀ����ҵ�ֶ����Թ۵��ļ�
% param.file_name = 'data/ȫ��ҵ����.xlsx';
% param.file_name = 'data/����300����.xlsx';
param.file_name = 'data/��֤500����.xlsx';
% param.file_name = 'data/����300���Զ�.xlsx';

% ��ָѡ�񣺻���300(HS300)����֤500(ZZ500)
param.targetIndex = 'ZZ500'; 

% �Ƿ�Ŀ��ָ����ѡ�ɣ�1=�ǣ�0=�񣩣�������ȫA��������
param.selectInIndex = 1;

% �������ϵ��
param.beta = 2;

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

% ��ȡ��ҵ�ֶ����Թ۵�
indusView = CalcRealIndusView(param,monthly_info);

% -------------------------------------------------------------------------
% �Ż��ز�
% -------------------------------------------------------------------------
% �������1��Ҳ������ҵ�۵����ǿ���
enhancePort = CalcEnhanceWeight(param,fullStockWeight,validStockMatrix,...
    forecastReturn,factorExpo,indusView,factorCov,specialCov);
[enhanceNav,bktestDates] = CalcStrategyNav(param,enhancePort,dailyClose,dailyDates);

% �������2��������ҵ���Ե�ѡ�����
neutralPort = CalcNeutralWeight(param,fullStockWeight,validStockMatrix,...
    forecastReturn,factorExpo,factorCov,specialCov);
[neutralNav,~] = CalcStrategyNav(param,neutralPort,dailyClose,dailyDates);

% �������3��Ҳ��ԭָ����ֵ
[~,bktestIndex] = ismember(bktestDates,dailyDates);
indexNav = indexClose(bktestIndex);
indexNav = indexNav ./ indexNav(1);

% ���㳬�ֵ
enhance2index = ret2tick(tick2ret(enhanceNav)-tick2ret(indexNav));
enhance2neutral = ret2tick(tick2ret(enhanceNav)-tick2ret(neutralNav));
neutral2index = ret2tick(tick2ret(neutralNav)-tick2ret(indexNav));

% ����ҵ��ָ��
perf = CalcStrategyPerf([enhanceNav,neutralNav,indexNav,enhance2index,enhance2neutral,neutral2index]);
perf = [{'';'��ǿ';'����';'ָ��';'��ǿ-ָ��';'��ǿ-����';'����-ָ��'},perf];

% -------------------------------------------------------------------------
% ��ͼ1����ǿ��������ָ����׼�ĳ������
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
legend({'��ǿ','ָ��','����'},'location','northwest');
set(gca,'FontSize',10,'FontWeight','bold','FontName','STFangSong','box','off');
xlim([1 length(dateStr)]);

% -------------------------------------------------------------------------
% ��ͼ2����ǿ��������������ϵĳ������
% -------------------------------------------------------------------------
% ��ͼ
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
legend({'��ǿ','����','����'},'location','northwest');
set(gca,'FontSize',10,'FontWeight','bold','FontName','STFangSong','box','off');
xlim([1 length(dateStr)]);

% -------------------------------------------------------------------------
% ��ͼ3��������������ָ����׼�ı���
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
legend({'����','ָ��','����'},'location','northwest');
set(gca,'FontSize',10,'FontWeight','bold','FontName','STFangSong','box','off');
xlim([1 length(dateStr)]);
