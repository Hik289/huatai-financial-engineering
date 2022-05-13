function indusView = CalcSimuViewByStock(param,dailyClose,fullStockWeight,factorExpo)
% -------------------------------------------------------------------------
% �������ܣ�������ҵ�ڳɷֹɼ��㣬����ָ��ʤ�ʵ���ҵ��չ۵�
% -------------------------------------------------------------------------

% ��ҵ���ӱ�¶������
indusFactorIndex = param.styleFactorNum+1:param.styleFactorNum+param.indusFactorNum;

% ��ʼ�����
indusView = nan(length(indusFactorIndex),length(param.month2day));

% ����ÿ������
for iMonth = param.beginMonth:(param.endMonth-1)
        
    % ��ȡ��������
    thisMonthEnd = param.month2day(1,iMonth);
    nextMonthEnd = param.month2day(1,iMonth+1);
    
    % ��ȡ��ǰ�����Ӧ����ҵ���ӱ�¶
    indusExpo = factorExpo(:,indusFactorIndex,thisMonthEnd);
    
    % ��ȡ��ǰ�����ϸ���Ȩ��
    stockWeight = fullStockWeight(:,iMonth);
    
    % ��ȡ������������
    stockReturn = dailyClose(:,nextMonthEnd)./dailyClose(:,thisMonthEnd)-1;
     
    % ���ݸ��������Ȩ�ϳ���ҵ��������
    panelIndusReturn =  nan(length(indusFactorIndex),1);
    for iIndus = 1:length(indusFactorIndex)
        % ����1����������Ŀ����ҵ
        condition1 = indusExpo(:,iIndus) == 1;
        % ����2������Ȩ�ط���
        condition2 = stockWeight > 0;
        % ����3��������������ǿ�
        condition3 = ~isnan(stockReturn);
        % ��ȡ��Ч�ĸ���Ȩ�ؼ���������
        valid = condition1 & condition2 & condition3;
        validWeight = stockWeight(valid);
        validReturn = stockReturn(valid);
        % �����Ȩ����
        panelIndusReturn(iIndus) = validWeight'*validReturn / sum(validWeight);
    end
    
    % ���ɽ���Ԥ��۵�
    view = [];
    while isempty(view)
        try 
            view = IndusViewGeneration(param,panelIndusReturn);
        catch
            % nothing to do
        end
    end
    indusView(:,iMonth) = view;
    
end

end

% -------------------------------------------------------------------------
% �������ܣ��������õľ���������ҵ��չ۵�
% 
% [����]
% param��           ȫ�ֲ����ṹ��
% panelIndusReturn����ҵָ�����µ���ʵ����
%
% [���]
% view�� ������ҵΪ1��������ҵ���Ϊ-1��������ҵ��Ϊ0��indusFactorNum * 1��
% -------------------------------------------------------------------------
function view = IndusViewGeneration(param,panelIndusReturn)

% Ԥ�⾫��
accuracy = param.accuracy;

% ��ҵ��Ȩ����
average_return = nanmedian(panelIndusReturn);

% �������������ҵ��λ�����࣬С�ڿ���
upwardIndus = find(panelIndusReturn >= average_return); % Ӧ���������ҵ
downwardIndus = find(panelIndusReturn < average_return);% Ӧ�����յ���ҵ

% ��ֵΪaccuracy��0-1����ֲ�,0��Ǳ�ʾԤ�����1��Ǳ�ʾԤ����ȷ
accuracy_vector = binornd(1,accuracy,param.unNeturalIndusNum,1);

% ��ʼ�����
view = nan(1,param.indusFactorNum);

% Ŀǰ����(��)����ҵ������
longIndusNum = 0;
shortIndusNum = 0; 

% �������ɶࡢ����ҵ�۵�
for iIndus = 1:param.unNeturalIndusNum

    % ���赱ǰ�۵�����ȷԤ��
    if accuracy_vector(iIndus)  
        if  longIndusNum < param.longIndusNum 
            % ����ʵ[��Ӯ]����ҵ����ѡһ������[����]������
            longIndus = upwardIndus(randperm(length(upwardIndus),1));
            view(longIndus) = 1;
            upwardIndus(upwardIndus == longIndus) = []; 
            longIndusNum = longIndusNum + 1;
        elseif shortIndusNum < param.shortIndusNum 
            % ����ʵ[����]����ҵ����ѡһ������[����]������
            shortIndus = downwardIndus(randperm(length(downwardIndus),1));
            view(shortIndus) = -1;
            downwardIndus(downwardIndus == shortIndus) = [];
            shortIndusNum = shortIndusNum + 1;
        end
    % ���赱ǰ�۵�Ԥ�����    
    else  
        if longIndusNum < param.longIndusNum 
            % ����ʵ[����]����ҵ����ѡһ������[����]������
            longIndus = downwardIndus(randperm(length(downwardIndus),1));
            view(longIndus) = 1;
            downwardIndus(downwardIndus == longIndus) = [];
            longIndusNum = longIndusNum+1;
        elseif shortIndusNum < param.shortIndusNum 
            % ����ʵ[��Ӯ]����ҵ����ѡһ������[����]������
            shortIndus = upwardIndus(randperm(length(upwardIndus),1));
            view(shortIndus) = -1;
            upwardIndus(upwardIndus == shortIndus) = [];
            shortIndusNum = shortIndusNum+1;
        end
    end    
end

% �������Ե���ҵ���Ϊ0
view(isnan(view)) = 0;

end

