function perf = CalcStrategyPerf(nav)
% -------------------------------------------------------------------------
% 计算组合净值的风险收益指标
% [输入]
% nav:  组合净值矩阵，其中每列代表一个组合
% [输出]
% perf: 每行代表一个组合，每列代表不同的指标
% -------------------------------------------------------------------------

% 获取净值矩阵大小
[dayNum,portNum] = size(nav);

% 初始化结果矩阵
perf = cell(portNum+1,4);
perf(1,:)={'年化收益率','年化波动率','夏普比率','最大回撤'};
           
% 计算每个组合的相关指标
for iPort = 1:portNum
    
    % 年化收益率
    perf{iPort+1,1} = nav(end,iPort)^(250/dayNum)-1;
    
    % 年化波动率
    perf{iPort+1,2} = std(nav(2:end,iPort)./nav(1:end-1,iPort))*sqrt(250);
    
    % 夏普比率
    perf{iPort+1,3} = perf{iPort+1,1} / perf{iPort+1,2};
    
    % 最大回撤
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


