% ---------------------------------------------------------------------
% ���������������ҵ�۵�ı��������Ԥ�⣬���׻��Ӱ���Ż�����
% ���Ľ��ۣ����������ҵ���ԣ���Ӱ�죬���������ҵƫ�룬���Ӱ��
% ---------------------------------------------------------------------
clear;clc;

% ---------------------------------------------------------------------
% ��������
% ---------------------------------------------------------------------
% ���ɻ�׼Ȩ��
w0 = [0.18;0.2;0.15;0.12;0.25;0.1];

% ����ԭʼ����Ԥ��
r = [0.05;0.015;0.02;0.04;0.035;-0.005];

% ������ҵ�۵�ı��������Ԥ��
delta = [0.02;0.02;0;0;-0.02;-0.02];

% ��ҵ��¶����
H = [1,0,0;...
     1,0,0;...
     0,1,0;...
     0,1,0;...
     0,0,1;...
     0,0,1];  

% ��ҵȨ��ƫ��
indusWeightUpLimit = 0.2;
indusWeightDownLimit = 0.05;

% ����ƫ������
stockWeightUpLimit = 0.05;

% �Ƿ񱣳���ҵ����
param.keepIndusNeutral = false;

% �Ƿ��������Ԥ��
param.adjust_return = false;

% ---------------------------------------------------------------------
% Ȩ��֮��Լ����Ȩ��ƫ��֮��Ϊ0��
% ---------------------------------------------------------------------
Aeq = ones(1,6);
beq = 0;

% ---------------------------------------------------------------------
% ����Ȩ��������Լ��
% ---------------------------------------------------------------------
uw = ones(6,1) * stockWeightUpLimit;
dw = -1 * w0;
    
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
panelIndusWeight = H' * w0;

% ��ȡ��ǰ������ҵԤ��۵�
if param.keepIndusNeutral
    panelIndustryView = [0;0;0];
else
    panelIndustryView = [1;0;-1];
end

% ����1���޳ɷֹ���ҵ
validStockInIndex = ones(6,1);
stockNumInIndus = sum(H(validStockInIndex==1,:));
emptyIndus = find(stockNumInIndus==0);
emptyIndusWeight = panelIndusWeight(emptyIndus);
emptyIndusWeightUpLimit = -1*emptyIndusWeight;
emptyIndusWeightDownLimit = -1*emptyIndusWeight;   
A2_0 = [H(:,emptyIndus)';-H(:,emptyIndus)'];
b2_0 = [emptyIndusWeightUpLimit;-emptyIndusWeightDownLimit];

% ��ȡ������Ч��ҵ�����Ұ���Ȩ�����¹�һ��
validIndus = setdiff(1:3,emptyIndus);
validIndusWeightOrig = panelIndusWeight(validIndus);
validIndusWeightUnit = validIndusWeightOrig / sum(validIndusWeightOrig);
validIndusWeightDelta = validIndusWeightUnit - validIndusWeightOrig;

% ����2��������ҵ
[neturalIndus,~,ia] = intersect(find(panelIndustryView==0),validIndus);
neturalIndusDelta = validIndusWeightDelta(ia);
meturalIndusLimit = zeros(length(neturalIndus),1);       
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
longIndusUpLimit = indusWeightUpLimit*ones(length(longIndus),1); 
longIndusDownLimit = indusWeightDownLimit*ones(length(longIndus),1);
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

% �ϲ�Լ��
A = [A2_0;A2_1;A2_2;A2_3];
b = [b2_0;b2_1;b2_2;b2_3];    


% ---------------------------------------------------------------------
% �Ż����  
% ---------------------------------------------------------------------   
options = optimset('Display','off');      
if param.adjust_return
    opt_w = linprog(-(r+delta),A,b,Aeq,beq,dw,uw,options);
else
    opt_w = linprog(-r,A,b,Aeq,beq,dw,uw,options);
end
    
final_w =  opt_w + w0
    
    
    