% -------------------------------------------------------------------------
% ����ָ��ʤ�ʵ�ģ���������ǿ�����µı���
% -------------------------------------------------------------------------
clear; clc; close all;
addpath('utils');   
dbstop if error

% -------------------------------------------------------------------------
% ���ò���(������ParamSet�ļ��ж���Ĭ�����ã���Ҫ�������ڱ��ű������¸�ֵ)
% -------------------------------------------------------------------------
% ��ȡĬ�ϲ���
param = ParamSet();

% ��ָѡ�񣺻���300(HS300)����֤500(ZZ500)
param.targetIndex = 'ZZ500';  

% ��ҵԤ�⾫��
param.accuracy = 0.7;

% ģ�����
param.simuNum = 20;

% ������ҵ�۵�ʱ�Ƿ����ָ���ڳɷֹɺϳ��������棬����ֱ����һ����ҵ
param.simuByStock = true;

% �Ƿ�Ŀ��ָ����ѡ�ɣ�1=�ǣ�0=�񣩣�������ȫA��������
param.selectInIndex = 0;   

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
% ��ȡ��׼���
% -------------------------------------------------------------------------
% ��ҵ�������
neutralPort = CalcNeutralWeight(param,fullStockWeight,validStockMatrix,...
                        forecastReturn,factorExpo,factorCov,specialCov);
[neutralNav,bktestDates] = CalcStrategyNav(param,neutralPort,dailyClose,dailyDates);

% ԭָ����ֵ
[~,bktestIndex] = ismember(bktestDates,dailyDates);
indexNav = indexClose(bktestIndex);
indexNav = indexNav ./ indexNav(1);

% -------------------------------------------------------------------------
% ���ؿ���ģ�����ȡƽ���������ǿ���Ա��֣������ϴ������ַ�����
% 1����ȡÿ��ģ��ľ�ֵ������ֵƽ������ҵ��ָ��
% 2��ֱ�Ӷ�ÿ��ģ���ҵ��ָ��ȡƽ��
% ����Ӧ�ò��õ�2��˼·����1��˼·ƽ����ֵ�����ǳ��ͣ����ʧ��
% -------------------------------------------------------------------------
% �洢ÿ��ģ���ҵ��ָ�꣬����Ϊ���������桢��������Բ��ԡ������ָ����׼
enhancePerf = nan(param.simuNum,12);

% ����ÿ��ģ��
for simu = 1:param.simuNum
    
    fprintf('    ��%d��ģ��\n',simu);

    % ������ҵԤ��۵�
    if param.simuByStock
        indusView = CalcSimuViewByStock(param,dailyClose,fullStockWeight,factorExpo);
    else
        indusView = CalcSimulateIndusView(param,indusClose);
    end
    
    % ���ɴ���ҵ�۵����ǿ���
    port = CalcEnhanceWeight(param,fullStockWeight,validStockMatrix,...
                forecastReturn,factorExpo,indusView,factorCov,specialCov);
    
    % �洢���Ծ�ֵ
    [nav,~] = CalcStrategyNav(param,port,dailyClose,dailyDates);
    
    % ������Ծ����������
    perf1 = CalcStrategyPerf(nav);
    
    % �����������������������������
    perf2 = CalcStrategyPerf(ret2tick(tick2ret(nav)-tick2ret(neutralNav)));
    
    % ������������ָ����׼����������
    perf3 = CalcStrategyPerf(ret2tick(tick2ret(nav)-tick2ret(indexNav)));
    
    % �洢���
    enhancePerf(simu,:) = cell2mat([perf1(2,:),perf2(2,:),perf3(2,:)]);

end

% ��ӡ���
cols = {'�껯����','�껯����','���ձ���','���س�',...
        '��������껯����','�껯����','��Ϣ����','���س�',...
        '���ָ���껯����','�껯����','��Ϣ����','���س�'};
perf = [cols; num2cell( mean(enhancePerf))];
