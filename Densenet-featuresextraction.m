clc
clear all
% Read the image to classify 
chemin='C:\Users\InfoMax\Desktop\NewCNNtentative\densenet\baseutilise\unique_UI_9677\';
mkdir('DensenetRico');
listing=dir(strcat(chemin,'*.jpg'));
% Access the trained model 
net = densenet201(); 
% See details of the architecture 
net.Layers 

for i = 1 : length(listing)
% Read the image to classify 
 ch=listing(i).name;
    % Read the image to classify 
     im=strcat(chemin,listing(i).name);
     name=string(listing(i).name);
     I = imread(im);
% See details of the architecture 
  layer = 'fc1000';
% Adjust size of the image 
sz = net.Layers(1).InputSize 
I = I(1:sz(1),1:sz(2),1:sz(3)); 
f = activations(net, I, layer, 'OutputAs', 'rows');
D=strcat('DensenetRico\',ch,'Densenetfeatures.mat');
save(D,'f');
end