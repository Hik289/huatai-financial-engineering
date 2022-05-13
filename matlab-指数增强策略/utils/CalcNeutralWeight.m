function port = CalcNeutralWeight(param,fullStockWeight,validStockMatrix,...
             forecastReturn,factorExpo,factorCov,specialCov)
% -------------------------------------------------------------------------
% ��ҵ������ϣ��Ż�����ΪȨ��ƫ�룬Լ��Ϊ��ҵ���ԣ��������κ���ҵ�ֶ��۵㣩
% -------------------------------------------------------------------------

% ��Ʊ�����ͽ�������
[stockNum,panelNum] = size(fullStockWeight);

% ��ʼ��Ȩ�ؽ��
port = nan(stockNum,1,panelNum);

% �ڲ�ѭ����ÿ��������
for iMonth = param.beginMonth:(param.endMonth-1)         

    % ---------------------------------------------------------------------
    % ����׼��
    % ---------------------------------------------------------------------
    % ��ȡ��������
    thisMonthEnd = param.month2day(1,iMonth);
    nextMonthBigen = thisMonthEnd + 1;

    % ���¸���������Ԥ��
    r = forecastReturn(1:stockNum,iMonth);

    % ���ձ�¶����
    X = factorExpo(:,:,thisMonthEnd);
    
    % �в�Э���������Э�������
    F = factorCov(:,:,thisMonthEnd);
    Delta = specialCov(:,thisMonthEnd).^2;   
    Delta = diag(Delta);
    
    % ��׼���Ȩ�أ�Ҳ����ָ���ɷֹ���չ��ȫ�г���Ľ������Ȩ�غ�Ϊ1
    W = fullStockWeight(:,iMonth);

    % ���ȫ�г���Ч�Ĺ�Ʊ
    validStockInMarket = (validStockMatrix(:,nextMonthBigen)==1) & ...
               (~isnan(sum(X,2))) & (~isnan(r)) & (~isnan(sum(Delta,2)));
                
    % �����ָ���ɷֹ�����Ч�Ĺ�Ʊ          
    validStockInIndex = double(validStockInMarket.*W>0);  
    
    % ��ȱʧֵ��Ϊ�㣬�����������ȫ�г���Ч��Ʊ���ʱ���޳���ȱʧֵ�Ĺ�Ʊ
    % �����������Щ��Ʊ�Ż�ʱ�����ǲ��ɽ��׵ģ�����������㲻��Ӱ��
    X(isnan(X))=0; X=double(X); 
    Delta(isnan(Delta)) = 0;
    r(isnan(r))=0;

    % ��ҵ���ӱ�¶����
    H = X(:,(param.styleFactorNum+1):(param.indusFactorNum+param.styleFactorNum));

    % ������ӱ�¶����                       
    S = X(:,1:param.styleFactorNum);

    % ��ֵ����
    Size = S(:,1);

    % ---------------------------------------------------------------------
    % Ȩ��֮��Լ����Ȩ��ƫ��֮��Ϊ0��
    % ---------------------------------------------------------------------
    Aeq = ones(1,stockNum);
    beq = 0;
   
    % ---------------------------------------------------------------------
    % ����Ȩ��������Լ��
    % ���ڿɽ��׹�Ʊ��Ȩ��ƫ������Ϊ-w_base��Ȩ��ƫ������ΪstockWeightLimit
    % ������Ч��Ʊ�������޾�Ϊ-w_base��Ҳ���ù�Ʊ����Ȩ��Ϊ��
    % ---------------------------------------------------------------------
    if param.selectInIndex 
        uw=-W; uw(validStockInIndex==1)=param.stockWeightUpLimit;
        dw=-W;
    else    
        uw=-W; uw(validStockInMarket==1)=param.stockWeightUpLimit; 
        dw=-W;
    end

    % ---------------------------------------------------------------------
    % ��ֵ��¶Լ��
    % ---------------------------------------------------------------------     
    A1 = [Size';-Size'];
    b1 = [param.sizeFactorLimit;param.sizeFactorLimit];

    % ---------------------------------------------------------------------
    % ��ҵ��¶Լ������Ϊ�����֣�
    % 1��������ҵ�ɷֹ�Ϊ�յ���ҵ��ֱ��ǿ�ƽ�����ҵ���Գֲ���Ϊ��
    % 2�������гɷֹɵ���ҵ��������ҵ����Լ��������Ϊ�˱�֤Ȩ�غ�Ϊ1��Լ��
    %    ��Щ��Ч��ҵ����Ҫ��������һ������
    % ����������Ҫ����Դ�ָ���ɷֹ���ѡ�ɵĳ�����ʵ����ȫ�г�ѡ�ɳ�����
    % ���ٻ�����ҵ�ڿɽ��׳ɷֹ�Ϊ��
    % ---------------------------------------------------------------------   
    % ��ȡ����ҵȨ��
    panelIndusWeight = H' * W;
    
    % ����1���޳ɷֹ���ҵ
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
    
    % ����2���гɷֹɵ���ҵ
    validIndus = setdiff(1:param.indusFactorNum,emptyIndus);
    validIndusWeightOrig = panelIndusWeight(validIndus);
    validIndusWeightUnit = validIndusWeightOrig / sum(validIndusWeightOrig);
    validIndusWeightDelta = validIndusWeightUnit - validIndusWeightOrig;
    validIndusWeightLimit = repmat(param.indusWeightNeutralLimit,length(validIndus),1);       
    A2_1 = [H(:,validIndus)';-H(:,validIndus)'];
    b2_1 = [validIndusWeightLimit+validIndusWeightDelta;...
            validIndusWeightLimit-validIndusWeightDelta];
    
    % �ϲ�Լ��
    A2 = [A2_0;A2_1];
    b2 = [b2_0;b2_1];
    
    % ---------------------------------------------------------------------
    % ָ����ǿ��Ʒ��ͬԼ����ָ���ڹ�ƱȨ�ر������80%
    % (w_base + x) * validStockInIndex >= 0.8
    % x * validStockInIndex >= 0.8 - w_base * validStockInIndex
    % -validStockInIndex * x <= w_base * validStockInIndex - 0.8
    % --------------------------------------------------------------------- 
    A3 = -1 * validStockInIndex';
    b3 = validStockInIndex' * W - 0.8;
    
    % �ϲ�����Լ��
    A = [A1;A2;A3];
    b = [b1;b2;b3];
    
    % ---------------------------------------------------------------------
    % �Ż����  
    % ---------------------------------------------------------------------   
    options = optimset('Display','off');
    if param.lambda == 0 
        % ֻ�������棬�������Թ滮�ķ�ʽ                
        opt_w = linprog(-r,A,b,Aeq,beq,dw,uw,options);
    else
        % ����Э���������ֵ����100�����������������Ż�Ŀ����ռ�ȹ��ͣ�
        F(isnan(F)) = 0;
        V = double(X*F*X'+ Delta);
        V = 100*(V+V')/2;       
        % �������ͷ��գ����ö��ι滮              
        opt_w = quadprog(param.lambda*V,-r,A,b,Aeq,beq,dw,uw,[],options);
    end

    % �������������������Ž⣬��ָ��Ȩ�����
    if isempty(opt_w)
        fprintf('        ��ǰ�����Ż�ʧ�ܣ�%d\n',iMonth);
        opt_w = W; 
    else
        opt_w = opt_w + W;
    end
    port(:,1,iMonth) = opt_w;    
end

% ������ʼ�·ݽض�
port(:,:,1:param.beginMonth-1) = [];

end