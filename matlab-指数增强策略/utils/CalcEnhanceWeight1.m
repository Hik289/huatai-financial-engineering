function port = CalcEnhanceWeight1(param,fullStockWeight,validStockMatrix,...
             forecastReturn,factorExpo,indusView,factorCov,specialCov)
% -------------------------------------------------------------------------
% ��ҵ��ǿ��ϣ��Ż�����ΪȨ��ƫ�룬Լ������ҵ�ֶ��۵������ǿ��
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
    
    % ��¼������������Ԥ��ĺ�����׼��
    
    
    % add by lic
    r(r==0) = nan;
    panelIndustryView = indusView(:,iMonth);
    X = factorExpo(:,:,thisMonthEnd);
    H = X(:,(param.styleFactorNum+1):(param.indusFactorNum+param.styleFactorNum));
    h = double(H) * panelIndustryView;
    sigma = std(r(~isnan(r)));
    r = param.beta*sigma*h + r; 

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
    % 2�������޹۵���ҵ��������ҵ����
    % 3�����ڿ�����ҵ��Ҫ��ǿ�Ƴ���
    % 4�����ڿ�����ҵ��Ҫ��ǿ�Ƶ��� 
    % ���ӵ�һ����Ҫ����Խ�ָ����ѡ�ɵĳ�����ȫ�г�ѡ�ɳ����£����ٻ����
    % ��ҵ�ڿɽ��׳ɷֹ�Ϊ�յ����
    % ---------------------------------------------------------------------   
    % ��ȡ����ҵȨ��
    panelIndusWeight = H' * W;
    
    % ��ȡ��ǰ������ҵԤ��۵�
    panelIndustryView = indusView(:,iMonth);  
%     panelIndustryView = zeros(30,1);
    
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
    
    % ��ȡ������Ч��ҵ�����Ұ���Ȩ�����¹�һ��
    validIndus = setdiff(1:param.indusFactorNum,emptyIndus);
    validIndusWeightOrig = panelIndusWeight(validIndus);
    validIndusWeightUnit = validIndusWeightOrig / sum(validIndusWeightOrig);
    validIndusWeightDelta = validIndusWeightUnit - validIndusWeightOrig;
    
    % ����2��������ҵ
    [neturalIndus,~,ia] = intersect(find(panelIndustryView==0),validIndus);
    neturalIndusDelta = validIndusWeightDelta(ia);
    meturalIndusLimit = repmat(param.indusWeightNeutralLimit,length(neturalIndus),1);       
    A2_1 = [H(:,neturalIndus)';-H(:,neturalIndus)'];
    b2_1 = [meturalIndusLimit+neturalIndusDelta;...
            meturalIndusLimit-neturalIndusDelta];
    
    % ����3����ͷ��ҵ
    % ���ڿ�����ҵ�����ܴ��ڿ�����Ϊ�յ����⣬����ĳ��ҵֻ��һ֧�ɽ��׸���
    % �����ʱҪ�����ҵ����1%�����������ƫ��Ҳ��1%����������ɿ�����Ϊ��
    % ��򵥵İ취���ǰѸ������Ȩ��ƫ��Ŵ󣬵���������ɸ�������󣬳�
    % ���������س�Ҳ���ױ��һ��������������ȫ�г�ѡ�ɳ����º��ٳ���
    [longIndus,~,ia] = intersect(find(panelIndustryView>0),validIndus);
    longIndusDelta = validIndusWeightDelta(ia);
    longIndusUpLimit = param.indusWeightUpLimit*ones(length(longIndus),1); 
    longIndusDownLimit = param.indusWeightDownLimit*ones(length(longIndus),1);
    A2_2 = [H(:,longIndus)';-H(:,longIndus)'];
    b2_2 = [longIndusUpLimit+longIndusDelta;...
            -longIndusDownLimit-longIndusDelta];
        
    % ����4����ͷ��ҵ
    % ���ڿ�����ҵҪ���䣬����ͨ�����׵��¿�����Ϊ�յ����⣬����A��ҵԭʼ
    % Ȩ��Ϊ0.5%��Լ����ԭ����������ҵ��ƫ��Ϊ[-4%,-1%],���谴��ԭʼ������
    % A��ҵ�ľ���Ȩ��Ϊ[-3.5%,-0.5%]֮�䣬��ʵ�����ǲ��������յģ����A��ҵ
    % ʵ���ϵĿ�ƫ���������[-0.5%,-0.5%],Ҳ�������ø���ҵ
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
          
    % �ϲ�Լ��
    A2 = [A2_0;A2_1;A2_2;A2_3];
    b2 = [b2_0;b2_1;b2_2;b2_3];
   
    % ����ҵԤ��۵����������������������r     
%     h = H * panelIndustryView;
%     r = param.beta*sigma*h + r;    
    
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