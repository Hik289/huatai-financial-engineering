% -------------------------------------------------------------------------
% �������ܣ���ȡ��Ч��Ʊ��Ǿ���Ҳ����ÿ���������޳��޷����׵Ĺ�Ʊ
%           - ���в���һ���
%           - ���еĹ�Ʊ
%           - T�ջ�����Ϊ0��
% -------------------------------------------------------------------------
% [����]
% basic_info����Ʊ������Ϣ
% daily_info����Ʊ��Ƶ��Ϣ
% [���]
% validStockMatrix����Ч��Ʊ���Ϊ1����Ч��Ʊ���Ϊnan��stockNum * dayNum��
% -------------------------------------------------------------------------
function validStockMatrix = CalcValidStockMatrix(basic_info,daily_info)

% ��Ʊ����
stockNum = length(basic_info.stock_code);   

% ���ڸ���   
dayNum = length(daily_info.dates);

% �������ڣ�stockNum * 1��
ipoDate = basic_info.ipo_date;      

% �������ڣ�stockNum * 1��
delistDate = basic_info.delist_date;       

% ��Ʊ������ stockNum * dayNum
turn = daily_info.turn;                       

% ��������ֵ�����ڸ���ֵ�Ĺ�Ʊ��Ϊ���ɽ��ף������޳�
thresholdLimite = 10^-8;     

% ��ʼ��
validStockMatrix = nan(stockNum,dayNum);                 

% ����ÿ֧��Ʊ
for iStock = 1:stockNum
    
    % ��������1���ʼ��Ϊ��Ч
    beginDateNo = sum(daily_info.dates<ipoDate(iStock)+365)+1; 
    
    % �����Ʊ���������������ǰһ�죬���߽�����������
    if ~isnan(delistDate(iStock))
        endDateNo = sum(daily_info.dates<delistDate(iStock));    
    else
        endDateNo=dayNum;
    end
    
    % ������Ч���
    validStockMatrix(iStock,beginDateNo:endDateNo)=1;

end

% �޳����ͻ��ֵ����ݣ�����ͣ�ƣ����߿�����ͣû�н�������
validStockMatrix(~(turn>thresholdLimite)) = nan;

end