clc
clear all
%# read some training data
% libSVM version_1
%   load('classes.mat');
%  load('C:\Users\InfoMax\Desktop\svm\feauters.mat');
 load('C:\Users\InfoMax\Desktop\svm\classemodif.mat');
 load('C:\Users\InfoMax\Desktop\svm\featuresgooglenetmodif.mat');
 d=double(intermed((1:100),:));
 cal=Classes(1:100);
% d=double(intermed);
% cal=Classes;
[~,~,labels]            = unique(cal);%Labels: 1/2/3
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
        model{k} = svmtrain(double(trainLabel==k),trainData,'-c 1 -g 0.25 -b 1');
    end

    %# get probability estimates of test instances using each model
      prob = zeros(size(testData,1),numLabels);
  
    for k=1:numLabels
      [~,~,p] = svmpredict(double(testLabel==k), testData, model{k}, '-b 1');

         prob(:,k) = p(:,model{k}.Label==1);    %# probability of class==k
    end
 %# predict the class with the highest probability
    [~,pred] = max(prob,[],2);
    acc = sum(pred == testLabel) ./ numel(testLabel)    %# accuracy  

       TP=0;
       FP=0;
       FN=0;
       TN=0;
      
     for k=1:numLabels
          TP(k) = length (find (testLabel == k &  pred == k ));
          TN(k) = length (find (testLabel ~= k &  pred ~= k ));
          FP(k) = length (find (testLabel ~= k & pred == k ));
          FN(k) = length (find (testLabel == k & pred ~= k ));
     end  
     TPS=0;
     FPS=0;
     FNS=0;
     TNS=0;
     for j = 1 : k
       TPS=TPS+ TP(j);
       FPS=FPS+FP(j);
       FNS=FNS+FN(j);
       TNS=TNS+TN(j);
     end
   Tpfold(fold)=TPS;
   FPfold(fold)=FPS;
   FNfold(fold)=FNS;
   TNfold(fold)=TNS;
   P(fold)=Tpfold(fold)/(Tpfold(fold)+FPfold(fold))
   R(fold)=Tpfold(fold)/(Tpfold(fold)+FNfold(fold))
%   C = confusionmat(testLabel, pred)                   %# confusion matrix                  
    AccuracyCV(fold)=acc(1)%mean(accuracy);
    [ConfusionMatrix{fold}, classes] = confusionmat(testLabel,pred);% categoryTest);  %# confusion matrix
    ConfusionMatrix{fold}= ConfusionMatrix{fold}
%     ./ repmat(sum(ConfusionMatrix{fold}, 2), 1, size(ConfusionMatrix{fold}, 2))
    disp(ConfusionMatrix{fold}); 
   
end

fprintf('Mean Accuracy 5CV :(%.2f%%)\n', mean(AccuracyCV));
confusionMatrix5Folds=(ConfusionMatrix{1}+ConfusionMatrix{2}+ConfusionMatrix{3}+ConfusionMatrix{4}+ConfusionMatrix{5})/5;
confusionMat= confusionMatrix5Folds;
% save confusionMatrix5Foldsgooglenet.mat confusionMat
plotconfusion(testLabel,pred)
imagesc(confusionMatrix5Folds);
title('Confusion Matrix Of GoogleNet features ')
saveas(gcf, 'ConfusionMatrixOfGoogleNetfeatures','jpg');
% TP=0;
% FP=0;
% FN=0;
% TN=0;
% for i = 1 : length(confusionMat)
% %  True positive: diagonal position, cm(x, x).
% TP=TP+confusionMat(i,i);
% % False positive: sum of column x (without main diagonal), sum(cm(:, x))-cm(x, x).
% FP=FP+(sum(confusionMat(:,i))-confusionMat(i,i));
% % False negative: sum of row x (without main diagonal), sum(cm(x, :), 2)-cm(x, x)
% FN=FN+sum(confusionMat(i, :), 2)-confusionMat(i, i);
% end
tpfinale=0;
fpfinale=0;
fnfinale=0;
tnfinale=0;
for v = 1 : 5
   tpfinale =tpfinale+Tpfold(v);
   fpfinale=fpfinale+FPfold(v);
   fnfinale=fnfinale+FNfold(v);
   tnfinale=tnfinale+TNfold(v);
end 
  tpfinale =tpfinale/5;
   fpfinale=fpfinale/5;
   fnfinale=fnfinale/5;
   tnfinale=tnfinale/5;
Precision=0;
Recall=0;
% Precision = TP / (TP+FP)
Precision = tpfinale *100/ (tpfinale+fpfinale) ; 
% Recall = TP / (TP+FN)
Recall = tpfinale *100/ (tpfinale+fnfinale);
%Accuracy = (tnfinale+tpfinale)/(tpfinale+tnfinale+fpfinale+fnfinale); 
 Accuracys = (tnfinale+tpfinale)*100/(tpfinale+tnfinale+fpfinale+fnfinale); 
% Precision=0;
% Recall=0;
% % Precision = TP / (TP+FP)
% Precision = TP / (TP+FP) ; 
% % Recall = TP / (TP+FN)
% Recall = TP / (TP+FN);

fprintf('True Positive :%.f \n',  tpfinale);
Tab{1,1}='True Positive';
Tab{1,2}= tpfinale;
fprintf('False Positive :%.f \n', fpfinale);
Tab{2,1}='False Positive';
Tab{2,2}=fpfinale;
fprintf('False negative :%.f \n', fnfinale);
Tab{3,1}='False negative';
Tab{3,2}=fnfinale;
fprintf('True negative :%.f \n', tnfinale);
Tab{4,1}='True negative';
Tab{4,2}=tnfinale;
fprintf('Precision :(% .2f%%)\n', Precision);
Tab{5,1}='Precision';
Tab{5,2}=Precision;
fprintf('Recall:(% .2f%%)\n', Recall);
Tab{6,1}='Recall';
Tab{6,2}=Recall;
fprintf('Mean Accuracy 5CV :(%.2f%%)\n', mean(AccuracyCV)*100);
Tab{7,1}='Mean Accuracy 5CV';
Tab{7,2}=mean(AccuracyCV)*100;
save metrique.mat Tab

% [X,Y]=perfcurve(testLabel,pred ,2);

[FPR, TPR, Thr, AUC, OPTROCPT]          = perfcurve(testLabel(:,1), prob(:,1), 1);
[FPR1, TPR1, Thr1, AUC1, OPTROCPT1]     = perfcurve(testLabel(:,1), prob(:,2), 1);
[FPR2, TPR2, Thr2, AUC2, OPTROCPT2]     = perfcurve(testLabel(:,1), prob(:,3), 1);
figure;
plot(FPR,TPR,FPR1,TPR1 , FPR2, TPR2)
% plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC curves for GoogleNet Features ')
saveas(gcf, 'ROCcurvesGoogleNet','jpg');