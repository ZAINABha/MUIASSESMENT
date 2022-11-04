clc
clear all
% Read the image to classify 
chemin='C:\Users\InfoMax\Desktop\basededonnéénouveau\Screenshots\';
mkdir('GoogleNetRicoNEWDATA');
listing=dir(strcat(chemin,'*.png'));
%traitment

for i = 1 : length(listing)
    ch=listing(i).name;
    % Read the image to classify 
     im=strcat(chemin,listing(i).name);
     name=string(listing(i).name);
     I = imread(im);
     % Load the trained model 
      net = googlenet;
     % See details of the architecture 
       layer = 'pool5-drop_7x7_s1';%1000
     % Adjust size of the image 
      %%Extract Image Features
         inputSize = net.Layers(1).InputSize;
         augimdsTrain = augmentedImageDatastore(inputSize(1:2),I );
         featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
         D=strcat('GoogleNetRicoNEWDATA\',ch,'GoogleNetfeatures.mat');
         save(D,'featuresTrain');  
end   