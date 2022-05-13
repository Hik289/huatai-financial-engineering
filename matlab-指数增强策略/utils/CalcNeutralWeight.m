function port = CalcNeutralWeight(param,fullStockWeight,validStockMatrix,...
             forecastReturn,factorExpo,factorCov,specialCov)
% -------------------------------------------------------------------------
% 行业中性组合（优化对象为权重偏离，约束为行业中性，不考虑任何行业轮动观点）
% -------------------------------------------------------------------------

% 股票数量和截面数量
[stockNum,panelNum] = size(fullStockWeight);

% 初始化权重结果
port = nan(stockNum,1,panelNum);

% 内层循环：每个截面期
for iMonth = param.beginMonth:(param.endMonth-1)         

    % ---------------------------------------------------------------------
    % 数据准备
    % ---------------------------------------------------------------------
    % 获取日期索引
    thisMonthEnd = param.month2day(1,iMonth);
    nextMonthBigen = thisMonthEnd + 1;

    % 下月个股收益率预测
    r = forecastReturn(1:stockNum,iMonth);

    % 风险暴露矩阵
    X = factorExpo(:,:,thisMonthEnd);
    
    % 残差协方差和因子协方差估计
    F = factorCov(:,:,thisMonthEnd);
    Delta = specialCov(:,thisMonthEnd).^2;   
    Delta = diag(Delta);
    
    % 基准组合权重，也即将指数成分股拓展到全市场后的结果，其权重和为1
    W = fullStockWeight(:,iMonth);

    % 标记全市场有效的股票
    validStockInMarket = (validStockMatrix(:,nextMonthBigen)==1) & ...
               (~isnan(sum(X,2))) & (~isnan(r)) & (~isnan(sum(Delta,2)));
                
    % 标记在指数成分股中有效的股票          
    validStockInIndex = double(validStockInMarket.*W>0);  
    
    % 将缺失值置为零，由于上面计算全市场有效股票标记时已剔除有缺失值的股票
    % 所以面针对这些股票优化时，都是不可交易的，因而这里置零不会影响
    X(isnan(X))=0; X=double(X); 
    Delta(isnan(Delta)) = 0;
    r(isnan(r))=0;

    % 行业因子暴露矩阵
    H = X(:,(param.styleFactorNum+1):(param.indusFactorNum+param.styleFactorNum));

    % 风格因子暴露矩阵                       
    S = X(:,1:param.styleFactorNum);

    % 市值因子
    Size = S(:,1);

    % ---------------------------------------------------------------------
    % 权重之和约束（权重偏离之和为0）
    % ---------------------------------------------------------------------
    Aeq = ones(1,stockNum);
    beq = 0;
   
    % ---------------------------------------------------------------------
    % 个股权重上下限约束
    % 对于可交易股票，权重偏离下限为-w_base，权重偏离上限为stockWeightLimit
    % 对于无效股票，上下限均为-w_base，也即该股票最终权重为零
    % ---------------------------------------------------------------------
    if param.selectInIndex 
        uw=-W; uw(validStockInIndex==1)=param.stockWeightUpLimit;
        dw=-W;
    else    
        uw=-W; uw(validStockInMarket==1)=param.stockWeightUpLimit; 
        dw=-W;
    end

    % ---------------------------------------------------------------------
    % 市值暴露约束
    % ---------------------------------------------------------------------     
    A1 = [Size';-Size'];
    b1 = [param.sizeFactorLimit;param.sizeFactorLimit];

    % ---------------------------------------------------------------------
    % 行业暴露约束，分为两部分：
    % 1、对于行业成分股为空的行业，直接强制将该行业绝对持仓置为零
    % 2、对于有成分股的行业，保持行业中性约束，但是为了保证权重和为1的约束
    %    这些有效行业都需要被动超配一定比例
    % 上述控制主要是针对纯指数成分股内选股的场景，实际上全市场选股场景下
    % 极少会有行业内可交易成分股为空
    % ---------------------------------------------------------------------   
    % 获取各行业权重
    panelIndusWeight = H' * W;
    
    % 操作1：无成分股行业
    if param.selectInIndex 
        stockNumInIndus = sum(H(validStockInIndex==1,:));
    else
        stockNumInIndus = sum(H(validStockInMarket,:));
    end
    emptyIndus = find(stockNumInIndus==0);
    emptyIndusWeight = panelIndusWeight(emptyIndus);
    emptyIndusWeightUpLimit = -1*emptyIndusWeight;
    emptyIndusWeightDownLimit = -1*emptyIndusWeight;   
    A2_0 = [H(:,emptyIndus)';-H(:,emptyIndus)'];
    b2_0 = [emptyIndusWeightUpLimit;-emptyIndusWeightDownLimit];
    
    % 操作2：有成分股的行业
    validIndus = setdiff(1:param.indusFactorNum,emptyIndus);
    validIndusWeightOrig = panelIndusWeight(validIndus);
    validIndusWeightUnit = validIndusWeightOrig / sum(validIndusWeightOrig);
    validIndusWeightDelta = validIndusWeightUnit - validIndusWeightOrig;
    validIndusWeightLimit = repmat(param.indusWeightNeutralLimit,length(validIndus),1);       
    A2_1 = [H(:,validIndus)';-H(:,validIndus)'];
    b2_1 = [validIndusWeightLimit+validIndusWeightDelta;...
            validIndusWeightLimit-validIndusWeightDelta];
    
    % 合并约束
    A2 = [A2_0;A2_1];
    b2 = [b2_0;b2_1];
    
    % ---------------------------------------------------------------------
    % 指数增强产品合同约束：指数内股票权重必须大于80%
    % (w_base + x) * validStockInIndex >= 0.8
    % x * validStockInIndex >= 0.8 - w_base * validStockInIndex
    % -validStockInIndex * x <= w_base * validStockInIndex - 0.8
    % --------------------------------------------------------------------- 
    A3 = -1 * validStockInIndex';
    b3 = validStockInIndex' * W - 0.8;
    
    % 合并线性约束
    A = [A1;A2;A3];
    b = [b1;b2;b3];
    
    % ---------------------------------------------------------------------
    % 优化求解  
    % ---------------------------------------------------------------------   
    options = optimset('Display','off');
    if param.lambda == 0 
        % 只考虑收益，采用线性规划的方式                
        opt_w = linprog(-r,A,b,Aeq,beq,dw,uw,options);
    else
        % 计算协方差矩阵（数值扩大100倍，避免最后风险在优化目标中占比过低）
        F(isnan(F)) = 0;
        V = double(X*F*X'+ Delta);
        V = 100*(V+V')/2;       
        % 兼顾收益和风险，采用二次规划              
        opt_w = quadprog(param.lambda*V,-r,A,b,Aeq,beq,dw,uw,[],options);
    end

    % 不存在满足条件的最优解，用指数权重替代
    if isempty(opt_w)
        fprintf('        当前截面优化失败：%d\n',iMonth);
        opt_w = W; 
    else
        opt_w = opt_w + W;
    end
    port(:,1,iMonth) = opt_w;    
end

% 按照起始月份截断
port(:,:,1:param.beginMonth-1) = [];

end