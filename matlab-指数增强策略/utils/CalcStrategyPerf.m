function perf = CalcStrategyPerf(nav)
% -------------------------------------------------------------------------
% ������Ͼ�ֵ�ķ�������ָ��
% [����]
% nav:  ��Ͼ�ֵ��������ÿ�д���һ�����
% [���]
% perf: ÿ�д���һ����ϣ�ÿ�д���ͬ��ָ��
% -------------------------------------------------------------------------

% ��ȡ��ֵ�����С
[dayNum,portNum] = size(nav);

% ��ʼ���������
perf = cell(portNum+1,4);
perf(1,:)={'�껯������','�껯������','���ձ���','���س�'};
           
% ����ÿ����ϵ����ָ��
for iPort = 1:portNum
    
    % �껯������
    perf{iPort+1,1} = nav(end,iPort)^(250/dayNum)-1;
    
    % �껯������
    perf{iPort+1,2} = std(nav(2:end,iPort)./nav(1:end-1,iPort))*sqrt(250);
    
    % ���ձ���
    perf{iPort+1,3} = perf{iPort+1,1} / perf{iPort+1,2};
    
    % ���س�
    max_drawdown = 0;
    for iDate=1:dayNum
        cur_drawdown = nav(iDate,iPort)/max(nav(1:iDate,iPort))-1;
        if cur_drawdown < max_drawdown
            max_drawdown = cur_drawdown;
        end
    end
    perf{iPort+1,4} = max_drawdown;   
    
end

end


