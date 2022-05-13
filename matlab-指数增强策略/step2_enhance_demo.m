% ---------------------------------------------------------------------
% 问题提出：基于行业观点改变个股收益预测，到底会否影响优化结论
% 核心结论：如果保持行业中性，则不影响，如果允许行业偏离，则会影响
% ---------------------------------------------------------------------
clear;clc;

% ---------------------------------------------------------------------
% 参数设置
% ---------------------------------------------------------------------
% 个股基准权重
w0 = [0.18;0.2;0.15;0.12;0.25;0.1];

% 个股原始收益预测
r = [0.05;0.015;0.02;0.04;0.035;-0.005];

% 基于行业观点改变个股收益预测
delta = [0.02;0.02;0;0;-0.02;-0.02];

% 行业暴露矩阵
H = [1,0,0;...
     1,0,0;...
     0,1,0;...
     0,1,0;...
     0,0,1;...
     0,0,1];  

% 行业权重偏离
indusWeightUpLimit = 0.2;
indusWeightDownLimit = 0.05;

% 个股偏离上限
stockWeightUpLimit = 0.05;

% 是否保持行业中性
param.keepIndusNeutral = false;

% 是否调整个股预测
param.adjust_return = false;

% ---------------------------------------------------------------------
% 权重之和约束（权重偏离之和为0）
% ---------------------------------------------------------------------
Aeq = ones(1,6);
beq = 0;

% ---------------------------------------------------------------------
% 个股权重上下限约束
% ---------------------------------------------------------------------
uw = ones(6,1) * stockWeightUpLimit;
dw = -1 * w0;
    
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
panelIndusWeight = H' * w0;

% 获取当前截面行业预测观点
if param.keepIndusNeutral
    panelIndustryView = [0;0;0];
else
    panelIndustryView = [1;0;-1];
end

% 操作1：无成分股行业
validStockInIndex = ones(6,1);
stockNumInIndus = sum(H(validStockInIndex==1,:));
emptyIndus = find(stockNumInIndus==0);
emptyIndusWeight = panelIndusWeight(emptyIndus);
emptyIndusWeightUpLimit = -1*emptyIndusWeight;
emptyIndusWeightDownLimit = -1*emptyIndusWeight;   
A2_0 = [H(:,emptyIndus)';-H(:,emptyIndus)'];
b2_0 = [emptyIndusWeightUpLimit;-emptyIndusWeightDownLimit];

% 获取所有有效行业，并且按照权重重新归一化
validIndus = setdiff(1:3,emptyIndus);
validIndusWeightOrig = panelIndusWeight(validIndus);
validIndusWeightUnit = validIndusWeightOrig / sum(validIndusWeightOrig);
validIndusWeightDelta = validIndusWeightUnit - validIndusWeightOrig;

% 操作2：中性行业
[neturalIndus,~,ia] = intersect(find(panelIndustryView==0),validIndus);
neturalIndusDelta = validIndusWeightDelta(ia);
meturalIndusLimit = zeros(length(neturalIndus),1);       
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
longIndusUpLimit = indusWeightUpLimit*ones(length(longIndus),1); 
longIndusDownLimit = indusWeightDownLimit*ones(length(longIndus),1);
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
shortIndusUpLimit = -indusWeightDownLimit*ones(length(shortIndus),1);
index = shortIndusWeight < indusWeightDownLimit;
shortIndusUpLimit(index) = -1*shortIndusWeight(index);
shortIndusDownLimit = -indusWeightUpLimit*ones(length(shortIndus),1);
index = shortIndusWeight < indusWeightUpLimit;
shortIndusDownLimit(index) = -1*shortIndusWeight(index); 
shortIndusDelta = validIndusWeightDelta(ia);
A2_3 = [H(:,shortIndus)';-H(:,shortIndus)'];
b2_3 = [shortIndusUpLimit+shortIndusDelta;...
        -shortIndusDownLimit-shortIndusDelta];

% 合并约束
A = [A2_0;A2_1;A2_2;A2_3];
b = [b2_0;b2_1;b2_2;b2_3];    


% ---------------------------------------------------------------------
% 优化求解  
% ---------------------------------------------------------------------   
options = optimset('Display','off');      
if param.adjust_return
    opt_w = linprog(-(r+delta),A,b,Aeq,beq,dw,uw,options);
else
    opt_w = linprog(-r,A,b,Aeq,beq,dw,uw,options);
end
    
final_w =  opt_w + w0
    
    
    