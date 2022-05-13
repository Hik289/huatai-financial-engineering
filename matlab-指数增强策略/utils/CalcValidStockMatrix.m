% -------------------------------------------------------------------------
% 函数功能：获取有效股票标记矩阵，也即在每个截面上剔除无法交易的股票
%           - 上市不满一年的
%           - 退市的股票
%           - T日换手率为0的
% -------------------------------------------------------------------------
% [输入]
% basic_info：股票基本信息
% daily_info：股票日频信息
% [输出]
% validStockMatrix：有效股票标记为1，无效股票标记为nan（stockNum * dayNum）
% -------------------------------------------------------------------------
function validStockMatrix = CalcValidStockMatrix(basic_info,daily_info)

% 股票个数
stockNum = length(basic_info.stock_code);   

% 日期个数   
dayNum = length(daily_info.dates);

% 上市日期（stockNum * 1）
ipoDate = basic_info.ipo_date;      

% 退市日期（stockNum * 1）
delistDate = basic_info.delist_date;       

% 股票换手率 stockNum * dayNum
turn = daily_info.turn;                       

% 换手率阈值：低于该阈值的股票视为不可交易，进行剔除
thresholdLimite = 10^-8;     

% 初始化
validStockMatrix = nan(stockNum,dayNum);                 

% 遍历每支股票
for iStock = 1:stockNum
    
    % 个股上市1年后开始记为有效
    beginDateNo = sum(daily_info.dates<ipoDate(iStock)+365)+1; 
    
    % 如果股票退市则计算至退市前一天，否者截至最新日期
    if ~isnan(delistDate(iStock))
        endDateNo = sum(daily_info.dates<delistDate(iStock));    
    else
        endDateNo=dayNum;
    end
    
    % 设置有效标记
    validStockMatrix(iStock,beginDateNo:endDateNo)=1;

end

% 剔除极低换手的数据（比如停牌，或者开盘涨停没有交易量）
validStockMatrix(~(turn>thresholdLimite)) = nan;

end