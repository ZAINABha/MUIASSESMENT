clc
clear all
%# read some training data

load('C:\Users\InfoMax\Desktop\DeepFeatures\RICO\featuresgooglenetmodif.mat');
load('C:\Users\InfoMax\Desktop\DeepFeatures\goodbadclasse.mat');

 a=intermmodif;
 b=classenew;
 [~,~,labels]            = unique(classenew);
[n,m]=size(intermmodif);
total_rows=(1:n);
original_features=a;
original_mark=labels;

ind = find(original_mark == 2);
% P = candidate points
P = original_features(ind ,:);
T = P';
% X = Complete Feature Vector
X = T;
% Finding the 5 positive nearest neighbours of all the positive blobs
I = fitcknn(T, original_mark, 'NumberOfNeighbours', 6);
I = I';
[r, c] = size(I);
S = [];
th=0.3;
for i=1:r
    for j=2:c
        index = I(i,j);
        new_P=P(i,:)+((P(index,:)-P(i,:))*rand);
        S = [S;new_P];
    end
end
original_features = [original_features;S];
[r c] = size(S);
mark = ones(r,1);
original_mark = [original_mark;mark];
train_incl = ones(length(original_mark), 1);
I = nearestneighbour(original_features', original_features', 'NumberOfNeighbours', 6);
I = I';
for j = 1:length(original_mark)
    neighbors = I(j, 2:6);
    len = length(find(original_mark(neighbors) ~= original_mark(j,1)));
    if(len >= 2)
        if(original_mark(j,1) == 1)
         train_incl(neighbors(original_mark(neighbors) ~= original_mark(j,1)),1) = 0;
        else
         train_incl(j,1) = 0;   
        end    
    end
end
final_features = original_features(train_incl == 1, :);
final_mark = original_mark(train_incl ==1, :);