function fullStockWeight = CalcFullStockWeight(index_data,basic_info)
% -------------------------------------------------------------------------
% 将指数成分股权重映射到全市场股票权重，比如中证500的权重分布对应的是500个
% 股票的权重，该函数目的就是映射到全市场3000多支股票上，权重值仍然不变
% [输入]
% index_data:      指数结构体，包含成分股列表，及其权重
% basic_info：     个股基本信息，包含全A股列表
% [输出]
% fullStockWeight：全市场权重矩阵（stockNum * dayNum）
% -------------------------------------------------------------------------

% 成分股列表
stockList = index_data.stock_list;   

% 成分股权重
stockWeight = index_data.stock_weight;   

% 成分股个数
[indexStockNum,panelNum] = size(stockList);

% 全市场股票列表
fullStockCode = basic_info.stock_code; 

% 计算各股票在指数中的权重，并映射到全股票矩阵中
fullStockWeight = zeros(length(fullStockCode),panelNum);                   
for iPanel = 1:panelNum
    
    % 早期指数尚未成立，成分股为空
    if sum(stockWeight(:,iPanel))==0
        continue;
    end
    
    % 获取有效的成分股索引
    goodLine = (stockWeight(:,iPanel)~=0);
    
    % 获取这些有效成分股在所有A股股票池中的位置索引
    validStock = nan(indexStockNum,1);  
    [~,validStock(goodLine)] = ismember(stockList(goodLine,iPanel),fullStockCode);
    
    % 可能存在一种极端情况：某只股票属于该股指，但是不在所有A股股票池中
    validStock(validStock==0) = nan; 
    validIndex = ~isnan(validStock);
    
    % 将权重重新归一化后赋值
    fullStockWeight(validStock(validIndex),iPanel) = ...
            stockWeight(validIndex,iPanel)/sum(stockWeight(validIndex,iPanel));    
end

end



