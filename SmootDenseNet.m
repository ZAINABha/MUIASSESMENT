clc
clear all
%# read some training data
load('C:\Users\InfoMax\Desktop\NewCNNtentative\densenet\DenseNetricofeat.mat');
load('C:\Users\InfoMax\Desktop\NewCNNtentative\densenet\goodbadclasse.mat');
 a=interdicenetrico;
 b=classenew;
 [~,~,labels]            = unique(classenew);
[n,m]=size(interdicenetrico);
total_rows=(1:n);
original_features=a;
original_mark=labels;
[balanced_SMOTE_DenseNet,final_labels_DenseNet]=SMOTE(original_features, original_mark);
save('balanced_SMOTE_DenseNet')
save('final_labels_DenseNet')