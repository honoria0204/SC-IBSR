
load('source_MI3DOR1_0.7.mat')
feats=source_feature;

source_feature1=normalize(feats); %normalize 10500
source_all_label=source_label+1;




load('target_MI3DOR1_0.7.mat')
target_feature1=normalize(target_feature);%normalize 3848  800
%target_label=t_label+1;
target_label=target_label+1;



disp('start to process Euclidean distance')
%final_adj=pdist2(target_feature,source_feature);  % source_feature size N*D(D dimension of feature)
final_adj=pdist2(source_feature1,target_feature1);
%load('final_adj.mat','final_adj')
disp('start to process performance')
% source_all_label=source_label(1:12:end)+1;


all_pingce=[];
class_label=unique(target_label);

[index2,rresult,pingce,pr_cure]=cross_performance(final_adj',target_label,source_all_label);

save('result_MI3DOR1.mat','pingce','pr_cure','source_all_label','target_label');
save('rresult.mat','rresult');
save('index2.mat','index2');



