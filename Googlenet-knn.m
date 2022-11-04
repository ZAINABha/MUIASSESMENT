clc
clear all
%# read some training data

 %load('C:\Users\InfoMax\Desktop\DeepFeatures\RICO\featuresgooglenetmodif.mat');
 
 load('C:\Users\InfoMax\Desktop\DeepFeatures\goodbadclasse.mat');
 d=double(intermmodif);
 cal=classenew;
[~,~,labels]            = unique(cal);   %Labels: 1/2/3
 data    = zscore(d);% Scale features
indices = crossvalind('kfold',labels,5);
confusionMatrix = cell(1,1);
errorMat = zeros(1,5);
for i = 1:5
train = (indices==i);
test = ~train;
modelknn = fitcknn(data(train,:),labels(train),'NumNeighbors',2,'Standardize',1);
y = predict(modelknn,data(test,:));

confusionMatrix{i} = confusionmat(labels(test),y);
acc = sum(y == labels(test)) ./ numel(labels(test)) 
acc1=acc*100
AccuracyCV(i)=acc(1)
end
% Calculate misclassification error

 fprintf('Mean Accuracy 5CV :(%.2f%%)\n', mean(AccuracyCV));
 confusionMatrix5Folds=(confusionMatrix{1}+confusionMatrix{2}+confusionMatrix{3}+confusionMatrix{4}+confusionMatrix{5})/5;
confusionMat= confusionMatrix5Folds;


TP=0;
FP=0;
FN=0;
TN=0;
for m=1:2
          TP = length (find (labels(test) == m & y == m ));
          TN = length (find (labels(test) ~= m & y ~= m ));
          FP = length (find (labels(test) ~= m & y == m ));
          FN = length (find (labels(test) == m & y ~= m ));     
         
end

Precision=0;
Recall=0;
% Precision = TP / (TP+FP)
Precision = TP*100 / (TP+FP) ; 
% Recall = TP / (TP+FN)
Recall = TP*100 / (TP+FN);

fprintf('True Positive :%.f \n', TP);
fprintf('True Negative :%.f \n', TN);
fprintf('False Positive :%.f \n', FP);
fprintf('False negative :%.f \n', FN);
fprintf('Precision :(%.2f%%)\n', Precision);
fprintf('Recall:(%.2f%%)\n', Recall);
fprintf('Mean Accuracy 5CV :(%.2f%%)\n', acc1);

