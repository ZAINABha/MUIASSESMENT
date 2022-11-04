clc
clear all
% liste=dir('C:\Users\InfoMax\Desktop\basededonnéénouveau\AlexNetRicoNEWDATA');
%     for i = 3 : length(liste)
%      ch=liste(i).name;
%   
%      feat=load(strcat('C:\Users\InfoMax\Desktop\basededonnéénouveau\AlexNetRicoNEWDATA\',ch));
%      val=struct2cell(feat);
%      val1=val{1,1};
%     end   
%      Alexnetfeatures(i-2,:)=val1(1:end);
% save Alexnetfeaturesnewdata.mat Alexnetfeatures
% liste=dir('C:\Users\InfoMax\Desktop\basededonnéénouveau\GoogleNetRicoNEWDATA');
%     for i = 3 : length(liste)
%      ch=liste(i).name;
%      feat=load(strcat('C:\Users\InfoMax\Desktop\basededonnéénouveau\GoogleNetRicoNEWDATA\',ch));
%      val=struct2cell(feat);
%      val1=val{1,1};
%      featuresgoogle(i-2,:)=val1(1:end);
%     end   
%     save GoogleNetfeatnewdata.mat featuresgoogle
%   save Googlenetredrawfeat.mat intergoogle
liste=dir('C:\Users\InfoMax\Desktop\basededonnéénouveau\VGG16RICONEWDATA');
    for i = 3 : length(liste)
     ch=liste(i).name;
     feat=load(strcat('C:\Users\InfoMax\Desktop\basededonnéénouveau\VGG16RICONEWDATA\',ch));
     val=struct2cell(feat);
     val1=val{1,1};
     featuresvgg16(i-2,:)=val1(1:end);
    end   
  save VGG16featnewdata.mat featuresvgg16