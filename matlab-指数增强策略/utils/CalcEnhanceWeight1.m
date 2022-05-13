function port = CalcEnhanceWeight1(param,fullStockWeight,validStockMatrix,...
             forecastReturn,factorExpo,indusView,factorCov,specialCov)
% -------------------------------------------------------------------------
% 行业增强组合（优化对象为权重偏离，约引入行业轮动观点进行增强）
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
    
    % 记录横截面个股收益预测的横截面标准差
    
    
    % add by lic
    r(r==0) = nan;
    panelIndustryView = indusView(:,iMonth);
    X = factorExpo(:,:,thisMonthEnd);
    H = X(:,(param.styleFactorNum+1):(param.indusFactorNum+param.styleFactorNum));
    h = double(H) * panelIndustryView;
    sigma = std(r(~isnan(r)));
    r = param.beta*sigma*h + r; 

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
    % 2、对于无观点行业，保持行业中性
    % 3、对于看多行业，要求强制超配
    % 4、对于看空行业，要求强制低配 
    % 单加第一步主要是针对仅指数内选股的场景，全市场选股场景下，很少会出现
    % 行业内可交易成分股为空的情况
    % ---------------------------------------------------------------------   
    % 获取各行业权重
    panelIndusWeight = H' * W;
    
    % 获取当前截面行业预测观点
    panelIndustryView = indusView(:,iMonth);  
%     panelIndustryView = zeros(30,1);
    
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
    
    % 获取所有有效行业，并且按照权重重新归一化
    validIndus = setdiff(1:param.indusFactorNum,emptyIndus);
    validIndusWeightOrig = panelIndusWeight(validIndus);
    validIndusWeightUnit = validIndusWeightOrig / sum(validIndusWeightOrig);
    validIndusWeightDelta = validIndusWeightUnit - validIndusWeightOrig;
    
    % 操作2：中性行业
    [neturalIndus,~,ia] = intersect(find(panelIndustryView==0),validIndus);
    neturalIndusDelta = validIndusWeightDelta(ia);
    meturalIndusLimit = repmat(param.indusWeightNeutralLimit,length(neturalIndus),1);       
    A2_1 = [H(:,neturalIndus)';-H(:,neturalIndus)'];
    b2_1 = [meturalIndusLimit+neturalIndusDelta;...
            meturalIndusLimit-neturalIndusDelta];
    
    % 操作3：多头行业
    % 对于看多行业，可能存在可行域为空的问题，比如某行业只有一支可交易个股
    % 如果此时要求该行业超配1%，而个股最大偏离也是1%，就容易造成可行域为空
    % 最简单的办法就是把个股最大权重偏离放大，但这容易造成跟踪误差变大，超
    % 额收益最大回撤也容易变大，一般而言这种情况在全市场选股场景下很少出现
    [longIndus,~,ia] = intersect(find(panelIndustryView>0),validIndus);
    longIndusDelta = validIndusWeightDelta(ia);
    longIndusUpLimit = param.indusWeightUpLimit*ones(length(longIndus),1); 
    longIndusDownLimit = param.indusWeightDownLimit*ones(length(longIndus),1);
    A2_2 = [H(:,longIndus)';-H(:,longIndus)'];
    b2_2 = [longIndusUpLimit+longIndusDelta;...
            -longIndusDownLimit-longIndusDelta];
        
    % 操作4：空头行业
    % 对于看空行业要低配，低配通常容易导致可行域为空的问题，比如A行业原始
    % 权重为0.5%，约束项原本对做空行业的偏离为[-4%,-1%],假设按照原始参数来
    % A行业的绝对权重为[-3.5%,-0.5%]之间，而实际上是不允许做空的，因此A行业
    % 实际上的可偏离区间就是[-0.5%,-0.5%],也即不配置该行业
    [shortIndus,~,ia] = intersect(find(panelIndustryView<0),validIndus);
    shortIndusWeight = validIndusWeightUnit(ia); 
    shortIndusUpLimit = -param.indusWeightDownLimit*ones(length(shortIndus),1);
    index = shortIndusWeight < param.indusWeightDownLimit;
    shortIndusUpLimit(index) = -1*shortIndusWeight(index);
    shortIndusDownLimit = -param.indusWeightUpLimit*ones(length(shortIndus),1);
    index = shortIndusWeight < param.indusWeightUpLimit;
    shortIndusDownLimit(index) = -1*shortIndusWeight(index); 
    shortIndusDelta = validIndusWeightDelta(ia);
    A2_3 = [H(:,shortIndus)';-H(:,shortIndus)'];
    b2_3 = [shortIndusUpLimit+shortIndusDelta;...
            -shortIndusDownLimit-shortIndusDelta];
          
    % 合并约束
    A2 = [A2_0;A2_1;A2_2;A2_3];
    b2 = [b2_0;b2_1;b2_2;b2_3];
   
    % 把行业预测观点叠加至个股收益上则修正r     
%     h = H * panelIndustryView;
%     r = param.beta*sigma*h + r;    
    
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