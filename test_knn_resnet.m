clc
clear all
%# read some training data
  load('C:\Users\InfoMax\Desktop\NewCNNtentative\resnet\balanced_SMOTE_resnet');
load('C:\Users\InfoMax\Desktop\NewCNNtentative\resnet\final_labels_resnet.mat');
 d=double(balanced_SMOTE);
 cal=final_labels;

[~,~,labels]            = unique(cal);   %Labels: 1/2/3
 data    = zscore(d);% Scale features
N = size(data, 1); % should be 36^2 = 1296
K = 5;
% create a vector that have K (=9) blocks of length N/K, such as [1 1 ... 1 2 2 ... 2 ... 9 9 ... 9 ]
folds = []; % again, sorry for the dynamic allocation
for i=1:K
    folds = [folds; repmat(i,floor(N/K), 1)];
end

accuracies = zeros(1, K);
for fold = 1:K
   testIds = find(folds==fold);
   trainIds = find(folds~=fold);
   % train your algorithm
   model = fitcknn(data(trainIds,:), labels(trainIds,:), 'NumNeighbors',3);
   
   % evaluate on the testing fold
   predictedLabels = predict(model,data(testIds,:));
R = confusionmat(predictedLabels,labels(testIds,:));
acc = sum(predictedLabels == labels(testIds,:)) ./ numel(labels(testIds,:)) 
acc1=acc*100
   accuracies(fold) = acc1
 ConfusionMatrix{fold}=R;
 disp(ConfusionMatrix{fold});
end
 fprintf('Mean Accuracy 5CV :(%.2f%%)\n', mean(accuracies));
 confusionMatrix5Folds=(ConfusionMatrix{1}+ConfusionMatrix{2}+ConfusionMatrix{3}+ConfusionMatrix{4}+ConfusionMatrix{5})/5;
confusionMat= confusionMatrix5Folds;