clc
clear all
%# read some training data
% libSVM version_1
 %load('C:\Users\InfoMax\Desktop\NewCNNtentative\densenet\DenseNetricofeat');
 %load('C:\Users\InfoMax\Desktop\NewCNNtentative\densenet\goodbadclasse.mat');
load('C:\Users\InfoMax\Desktop\NewCNNtentative\densenet\balanced_SMOTE_DenseNet');
load('C:\Users\InfoMax\Desktop\NewCNNtentative\densenet\final_labels_DenseNet');

d=double(balanced_SMOTE_DenseNet);
cal=final_labels_DenseNet;

[~,~,labels]            = unique(cal);   %Labels: 1/2/3
 data    = zscore(d);% Scale features

n                       = size(d,1);
numLabels               = max(labels);

ns = floor(n/5);
for fold=1:5
    if fold==1
        testindices= ((fold-1)*ns+1):fold*ns;
        trainindices = fold*ns+1:n;
    else
        if fold==5
            testindices= ((fold-1)*ns+1):n;
            trainindices = 1:(fold-1)*ns;
        else
            testindices= ((fold-1)*ns+1):fold*ns;
            trainindices = [1:(fold-1)*ns,fold*ns+1:n];
         end
    end
    % use testindices only for testing and train indices only for testing
    trainLabel = labels(trainindices);
    trainData = data(trainindices,:);
    testLabel = labels(testindices);
    testData = data(testindices,:);
    
    %# train one-against-all models
    model = cell(numLabels,1);
    for k=1:numLabels
        model{k} = svmtrain(double(trainLabel==k), trainData, '-t 0 -c 1 -b 1');
    end

    %#0.2 get probability estimates of test instances using each model
      prob = zeros(size(testData,1),numLabels);
  
    for k=1:numLabels
        [~,~,p] = svmpredict(double(testLabel==k), testData, model{k}, '-b 1');
        prob(:,k) = p(:,model{k}.Label==1);    %# probability of class==k
    end
 %# predict the class with the highest probability
    [~,pred] = max(prob,[],2);
      acc = sum(pred == testLabel) ./ numel(testLabel)    %# accuracy
%       C = confusionmat(testLabel, pred)                   %# confusion matrix                  
    AccuracyCV(fold)=acc(1)%mean(accuracy);
   [ConfusionMatrix{fold}, classes] = confusionmat(testLabel',pred);% categoryTest);  %# confusion matrix
%    ConfusionMatrix{fold}= ConfusionMatrix{fold}./ repmat(sum(ConfusionMatrix{fold}, 2), 1, size(ConfusionMatrix{fold}, 2))
    disp(ConfusionMatrix{fold});
end

fprintf('Mean Accuracy 5CV :(%.2f%%)\n', mean(AccuracyCV));
confusionMatrix5Folds=(ConfusionMatrix{1}+ConfusionMatrix{2}+ConfusionMatrix{3}+ConfusionMatrix{4}+ConfusionMatrix{5})/5;
confusionMat= confusionMatrix5Folds;
% TP=confusionMat(1,1);
% FP=confusionMat(1,2);
% FN=confusionMat(2,1);
% TN=confusionMat(2,2);

% imagesc(confusionMatrix5Folds);
% title('Confusion Matrix Of VGG16 features ')
% saveas(gcf, 'ConfusionMatrixOfVGG16features','jpg');

TP=0;
FP=0;
FN=0;
TN=0;
for i = 1 : length(confusionMat)
%  True positive: diagonal position, cm(x, x).
TP=TP+confusionMat(i,i);
% False positive: sum of column x (without main diagonal), sum(cm(:, x))-cm(x, x).
FP=FP+(sum(confusionMat(:,i))-confusionMat(i,i));
% False negative: sum of row x (without main diagonal), sum(cm(x, :), 2)-cm(x, x)
FN=FN+sum(confusionMat(i, :), 2)-confusionMat(i, i);
end


% for m=1:2
%           TP = length (find (testLabel' == m & pred == m ));
%           TN = length (find (testLabel' ~= m & pred ~= m ));
%           FP = length (find (testLabel' ~= m & pred == m ));
%           FN = length (find (testLabel' == m & pred ~= m ));     
%          
% end 
Precision=0;
Recall=0;
% Precision = TP / (TP+FP)
Precision = TP*100 / (TP+FP) ; 
% Recall = TP / (TP+FN)
Recall = TP*100 / (TP+FN);

% fprintf('True Positive :%.f \n', TP);
% Tab{1,1}='True Positive';
% Tab{1,2}=TP;
% fprintf('False Positive :%.f \n', FP);
% Tab{2,1}='False Positive';
% Tab{2,2}=FP;
% fprintf('False negative :%.f \n', FN);
% Tab{3,1}='False negative';
% Tab{3,2}=FN;
fprintf('Precision :(%.2f%%)\n', Precision);
% Tab{4,1}='Precision';
% Tab{4,2}=Precision;
fprintf('Recall:(%.2f%%)\n', Recall);
% Tab{5,1}='Recall';
% Tab{5,2}=Recall;
fprintf('Mean Accuracy 5CV :(%.2f%%)\n', mean(AccuracyCV)*100);
% Tab{6,1}='Mean Accuracy 5CV';
% Tab{6,2}=mean(AccuracyCV);
% save metrique.mat Tab
% [X,Y,T,AUC]=perfcurve(testLabel,pred ,1);
% figure;plot(X,Y)
% title('ROC curves for VGG16 Features ')
% saveas(gcf, 'ROCcurvesVGG16','jpg');