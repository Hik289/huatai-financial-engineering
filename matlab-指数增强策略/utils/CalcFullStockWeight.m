function fullStockWeight = CalcFullStockWeight(index_data,basic_info)
% -------------------------------------------------------------------------
% ��ָ���ɷֹ�Ȩ��ӳ�䵽ȫ�г���ƱȨ�أ�������֤500��Ȩ�طֲ���Ӧ����500��
% ��Ʊ��Ȩ�أ��ú���Ŀ�ľ���ӳ�䵽ȫ�г�3000��֧��Ʊ�ϣ�Ȩ��ֵ��Ȼ����
% [����]
% index_data:      ָ���ṹ�壬�����ɷֹ��б�����Ȩ��
% basic_info��     ���ɻ�����Ϣ������ȫA���б�
% [���]
% fullStockWeight��ȫ�г�Ȩ�ؾ���stockNum * dayNum��
% -------------------------------------------------------------------------

% �ɷֹ��б�
stockList = index_data.stock_list;   

% �ɷֹ�Ȩ��
stockWeight = index_data.stock_weight;   

% �ɷֹɸ���
[indexStockNum,panelNum] = size(stockList);

% ȫ�г���Ʊ�б�
fullStockCode = basic_info.stock_code; 

% �������Ʊ��ָ���е�Ȩ�أ���ӳ�䵽ȫ��Ʊ������
fullStockWeight = zeros(length(fullStockCode),panelNum);                   
for iPanel = 1:panelNum
    
    % ����ָ����δ�������ɷֹ�Ϊ��
    if sum(stockWeight(:,iPanel))==0
        continue;
    end
    
    % ��ȡ��Ч�ĳɷֹ�����
    goodLine = (stockWeight(:,iPanel)~=0);
    
    % ��ȡ��Щ��Ч�ɷֹ�������A�ɹ�Ʊ���е�λ������
    validStock = nan(indexStockNum,1);  
    [~,validStock(goodLine)] = ismember(stockList(goodLine,iPanel),fullStockCode);
    
    % ���ܴ���һ�ּ��������ĳֻ��Ʊ���ڸù�ָ�����ǲ�������A�ɹ�Ʊ����
    validStock(validStock==0) = nan; 
    validIndex = ~isnan(validStock);
    
    % ��Ȩ�����¹�һ����ֵ
    fullStockWeight(validStock(validIndex),iPanel) = ...
            stockWeight(validIndex,iPanel)/sum(stockWeight(validIndex,iPanel));    
end

end



