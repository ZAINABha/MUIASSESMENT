clc
clear all
%# read some training data
load('C:\Users\InfoMax\Desktop\NewCNNtentative\resnet\RESNETRico');
load('C:\Users\InfoMax\Desktop\NewCNNtentative\resnet\goodbadclasse.mat');
 a=interresnetrico;
 b=classenew;
 [~,~,labels]            = unique(classenew);
[n,m]=size(interresnetrico);
total_rows=(1:n);
original_features=a;
original_mark=labels;
[balanced_SMOTE,final_labels]=SMOTE(original_features, original_mark);
save('balanced_SMOTE')
save('final_labels')