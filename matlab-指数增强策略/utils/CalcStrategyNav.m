function [bktestNav,bktestDates] = CalcStrategyNav(param,port,dailyClose,dailyDates)
% -------------------------------------------------------------------------
% ������ϵľ�ֵ����
% [����]
% param:        �����ṹ��
% port��        ���Ȩ�أ�stockNum * portNum * panelNum��
% close��       �ز�������ʲ����̼ۣ�stockNum * dayNum��
% dates:        �ز������������У�1 * dayNum��
% [���]
% nav��         ����Ͼ�ֵ���ƣ�dayNum * portNum��
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% ��ȡ����
% -------------------------------------------------------------------------
% ���ά�Ⱥͽ���ά��
[~,portNum,~] = size(port);

% ���������ѣ����ߣ�
fee = param.fee;

% -------------------------------------------------------------------------
% ��ȡ�ز��������������
% -------------------------------------------------------------------------
% �ز⿪ʼ��
firstday = param.month2day(1,param.beginMonth)+1;

% �ز������
endday = param.month2day(1,param.endMonth);

% ��������(ÿ���³�)
refreshDates = dailyDates(param.month2day(1,param.beginMonth:(param.endMonth-1))+1);

% ��ȡ�ز���������̼ۼ���������
bktestClose = dailyClose(:,firstday:endday);
bktestDates = dailyDates(firstday:endday);
dayNum = length(bktestDates);

% -------------------------------------------------------------------------
% ������Ͼ�ֵ
% -------------------------------------------------------------------------
% ��ʼ�����
bktestNav = nan(dayNum, portNum);   

% ����������ϣ����ɾ�ֵ����
for iPort = 1:portNum
    
    % ��ȡ��ϵ���ʷȨ�ؾ���
    portWeight = squeeze(port(:,iPort,:));
    
    % ---------------------------------------------------------------------
    % �ز�������ʼ�վ��ǵ�һ�������գ����ﹹ����ʼ��λ
    % ---------------------------------------------------------------------   
    % ��ȡ���µĳֲ�Ȩ�أ�ע������Ȩ���Ѿ��ǹ�һ���Ľ��
    refreshIndex = 1;
    refresh_w = portWeight(:,1);
    
    % ������ϣ���ֵ�۳�������
    bktestNav(1,iPort) = 1 - fee;
    last_portfolio = (1- fee) .* refresh_w;

    % ---------------------------------------------------------------------
    % ����Ƶ���������¾�ֵ���������ո�����λ
    % ---------------------------------------------------------------------
    for iDate = 2:dayNum
       
        % 1��ִ�о�ֵ���£��ֿղֺͲ��ղ�����  
        if nansum(last_portfolio) == 0
            bktestNav(iDate,iPort) = bktestNav(iDate-1,iPort);
        else
            last_portfolio = bktestClose(:,iDate) ./ bktestClose(:,iDate-1) .* last_portfolio;
            bktestNav(iDate,iPort) = nansum(last_portfolio);
        end

        % 2���жϵ�ǰ�����Ƿ�Ϊ�µĵ����գ�������е���
        if ismember(bktestDates(iDate),refreshDates)           
            
            % ��¼����ǰ��Ȩ�طֲ���Ҳ�������²�λ��һ��
            if sum(last_portfolio) == 0
                last_w = zeros(size(last_portfolio));
            else
                last_w = last_portfolio ./ nansum(last_portfolio);
                last_w(isnan(last_w)) = 0;
            end
            
            % ��ȡ���µ�Ȩ�طֲ���ע�������Ȩ�طֲ��ǹ�һ���Ľ��
            refreshIndex = refreshIndex + 1;   
            refresh_w = portWeight(:,refreshIndex);
            
            % ����ǰ��Ȩ�ز����㻻���ʣ���������ֵ
            turnover = sum(abs(refresh_w - last_w));
            bktestNav(iDate,iPort) = bktestNav(iDate,iPort) * (1 - turnover * fee);
            last_portfolio = bktestNav(iDate,iPort) .* refresh_w;
            
        end
    end
end

end