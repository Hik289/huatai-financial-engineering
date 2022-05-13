function [bktestNav,bktestDates] = CalcStrategyNav(param,port,dailyClose,dailyDates)
% -------------------------------------------------------------------------
% 计算组合的净值走势
% [输入]
% param:        参数结构体
% port：        组合权重（stockNum * portNum * panelNum）
% close：       回测区间的资产收盘价（stockNum * dayNum）
% dates:        回测区间日期序列（1 * dayNum）
% [输出]
% nav：         各组合净值走势（dayNum * portNum）
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% 获取参数
% -------------------------------------------------------------------------
% 组合维度和截面维度
[~,portNum,~] = size(port);

% 交易手续费（单边）
fee = param.fee;

% -------------------------------------------------------------------------
% 获取回测区间与调仓日期
% -------------------------------------------------------------------------
% 回测开始日
firstday = param.month2day(1,param.beginMonth)+1;

% 回测结束日
endday = param.month2day(1,param.endMonth);

% 调仓日期(每月月初)
refreshDates = dailyDates(param.month2day(1,param.beginMonth:(param.endMonth-1))+1);

% 获取回测区间的收盘价及日期序列
bktestClose = dailyClose(:,firstday:endday);
bktestDates = dailyDates(firstday:endday);
dayNum = length(bktestDates);

% -------------------------------------------------------------------------
% 计算组合净值
% -------------------------------------------------------------------------
% 初始化结果
bktestNav = nan(dayNum, portNum);   

% 遍历各个组合，生成净值走势
for iPort = 1:portNum
    
    % 获取组合的历史权重矩阵
    portWeight = squeeze(port(:,iPort,:));
    
    % ---------------------------------------------------------------------
    % 回测区间起始日就是第一个换仓日，这里构建初始仓位
    % ---------------------------------------------------------------------   
    % 获取最新的持仓权重，注意这里权重已经是归一化的结果
    refreshIndex = 1;
    refresh_w = portWeight(:,1);
    
    % 建仓完毕，净值扣除手续费
    bktestNav(1,iPort) = 1 - fee;
    last_portfolio = (1- fee) .* refresh_w;

    % ---------------------------------------------------------------------
    % 按日频遍历，更新净值，遇调仓日更换仓位
    % ---------------------------------------------------------------------
    for iDate = 2:dayNum
       
        % 1、执行净值更新，分空仓和不空仓情形  
        if nansum(last_portfolio) == 0
            bktestNav(iDate,iPort) = bktestNav(iDate-1,iPort);
        else
            last_portfolio = bktestClose(:,iDate) ./ bktestClose(:,iDate-1) .* last_portfolio;
            bktestNav(iDate,iPort) = nansum(last_portfolio);
        end

        % 2、判断当前日期是否为新的调仓日，是则进行调仓
        if ismember(bktestDates(iDate),refreshDates)           
            
            % 记录调仓前的权重分布，也即将最新仓位归一化
            if sum(last_portfolio) == 0
                last_w = zeros(size(last_portfolio));
            else
                last_w = last_portfolio ./ nansum(last_portfolio);
                last_w(isnan(last_w)) = 0;
            end
            
            % 获取最新的权重分布，注意这里的权重分布是归一化的结果
            refreshIndex = refreshIndex + 1;   
            refresh_w = portWeight(:,refreshIndex);
            
            % 根据前后权重差别计算换手率，并调整净值
            turnover = sum(abs(refresh_w - last_w));
            bktestNav(iDate,iPort) = bktestNav(iDate,iPort) * (1 - turnover * fee);
            last_portfolio = bktestNav(iDate,iPort) .* refresh_w;
            
        end
    end
end

end