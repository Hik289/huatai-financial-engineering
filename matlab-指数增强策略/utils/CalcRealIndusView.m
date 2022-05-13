function indusView = CalcRealIndusView(param,monthly_info)
% -------------------------------------------------------------------------
% 函数功能：获取真实的行业轮动观点
% 
% [输入]
% param：        全局参数结构体
% monthly_info:  本地月频数据结构体
% indusWeight:   行业权重矩阵(indusNum * panelNum)
%
% [输出]
% view： 看多行业为1，看空行业标记为-1，中性行业记为0（indusFactorNum * 1）
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% step1：先按照文件输入读取原始数据
% -------------------------------------------------------------------------
% 读取文件
[~,~,raw] = xlsread(param.file_name);
dates = datenum(raw(2:end,1));
data = cell2mat(raw(2:end,2:end));
indus = raw(1,2:end);

% 本地数据中行业的排布顺序
local_indus = {'煤炭','交通运输','房地产','电力及公用事业','机械',...
                '电力设备及新能源','有色金属','基础化工','商贸零售',...
                '建筑','轻工制造','综合','医药','纺织服装','食品饮料',...
                '家电','汽车','电子','建材','消费者服务','石油石化',...
                '国防军工','农林牧渔','钢铁','通信','计算机','非银行金融',...
                '传媒','银行','综合金融'};


% 将观点填充至与本地日期相同长度
[flag1,index1] = ismember(dates,monthly_info.dates);
if any(flag1==0)
    error('行业轮动观点的日期与本地日期不匹配');
end
indusView = nan(length(monthly_info.dates),length(local_indus));
indusView(index1,:) = data;

% 将行业顺序调整至与本地对齐
[flag2,index2] = ismember(local_indus,indus);
if any(flag2==0)
    error('行业轮动文件的行业名称与本地行业名称不匹配');
end
indusView = indusView(:,index2);

% 转置
indusView = indusView';


end
