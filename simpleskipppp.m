%% Automatic Classification of Glaucoma Stages From Fundus Images using R-CNN & RFCN
clc
clear all
close all
imds = imageDatastore('C:\Users\MEBIN\Desktop\fundusmathworks\Database',...
    'IncludeSubfolders',true,... 
    'LabelSource','foldernames');
[Data,testData]= splitEachLabel(imds,0.8,'randomize');
% Training files 
[trainData] =Data;
layers = [
    imageInputLayer([128 200 3],'Name','input') %% resize
    convolution2dLayer(5,16,'Padding','same','Name','conv_1')
    batchNormalizationLayer('Name','BN_1')% MASKING
    reluLayer('Name','relu_1')
    convolution2dLayer(3,32,'Padding','same','Stride',2,'Name','conv_2')
    batchNormalizationLayer('Name','BN_2')
    reluLayer('Name','relu_2')
    convolution2dLayer(3,32,'Padding','same','Name','conv_3')
    batchNormalizationLayer('Name','BN_3')
    reluLayer('Name','relu_3')
    additionLayer(3,'Name','add')
    averagePooling2dLayer(3,'Stride',2,'Name','avpool')
    fullyConnectedLayer(3,'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classOutput')];

% Create a layer graph from the layer array. layerGraph connects all the layers in layers sequentially. Plot the layer graph.
lgraph = layerGraph(layers);
figure
plot(lgraph)

% Create the 1-by-1 convolutional layer and add it to the layer graph. Specify the number of convolutional filters and the stride so that the activation size matches the activation size of the 'relu_3' layer. This arrangement enables the addition layer to add the outputs of the 'skipConv' and 'relu_3' layers. To check that the layer is in the graph, plot the layer graph.
skipConv = convolution2dLayer(2,32,'Stride',2,'Name','skipConv');
lgraph = addLayers(lgraph,skipConv);
figure
plot(lgraph)

% Create the shortcut connection from the 'relu_1' layer to the 'add' layer. Because you specified two as the number of inputs to the addition layer when you created it, the layer has two inputs named 'in1' and 'in2'. The 'relu_3' layer is already connected to the 'in1' input. Connect the 'relu_1' layer to the 'skipConv' layer and the 'skipConv' layer to the 'in2' input of the 'add' layer. The addition layer now sums the outputs of the 'relu_3' and 'skipConv' layers. To check that the layers are connected correctly, plot the layer graph.
lgraph = connectLayers(lgraph,'relu_1','skipConv');
lgraph = connectLayers(lgraph,'skipConv','add/in2');
 lgraph = connectLayers(lgraph,'relu_2','add/in3');
%  
figure
plot(lgraph);

disp('%%%%----------RCNN CLASSIFICATION---------------%%%');

%% RCNN TRAINING USING ADAM OPTIMIZER (ADAM)
options1 = trainingOptions('adam', ...
    'MiniBatchSize',128, ...
    'MaxEpochs',20, ... % was 6
    'ValidationFrequency',5, ...
    'InitialLearnRate',1e-4,'Plots','training-progress');
% % network training 
 [convnet, traininfo] = trainNetwork(trainData,lgraph,options1);
inp=input('Enter input :');
    I7 = imread(inp);
    figure(1),imshow(I7)
%% PREPROCESSING
% Im1= imread(filename);
I = im2double(rgb2gray(I7));
figure,imshow(I);title('Input Grey Image')
% Rotating Image
I = imrotate(I, 45);
figure,imshow(I);title('Input Rotate Image')
% Noise Removal
I1 = imnoise(rgb2gray(I7),'salt & pepper',0.02);% CLIP LIMIT
subplot(1,2,1),imshow(I1);
title('Noise addition');
K = medfilt2(I1);
subplot(1,2,2),imshow(K);
title('Noise removal using median filter');
tic;
%% EMPERICAL MODE DECOMPOSOTION
l =1;
while (l < 4)
   %% MAX AND MIN REGIONS
        lmax = imregionalmax(I);
%         figure,imshow(lmin);title('imregionalmax Image')
        lmin = imregionalmax(-I);
% figure,imshow(lmin);title('imregionalmax -Image')
        Vmax = I.*lmax;
        Vmin = I.*lmin;

        [m, n]  = size(I);

        [Xp, Yp] = meshgrid(1:n, 1:m);

        [X, Y] = find(Vmax ~=0);
        Zmax = Vmax(Vmax ~=0);
        Vpmax = griddata(Y, X, Zmax, Xp, Yp, 'cubic');
         Vpmax(isnan(Vpmax))  = 0;  
        [X, Y] = find(Vmin ~=0);
        Zmin = Vmin(Vmin ~=0);
        Vpmin  = griddata(Y, X, Zmin,  Xp, Yp, 'cubic');
        Vpmin(isnan(Vpmin))  = 0;  
        m = (Vpmax + Vpmin)/2;
        h = (I -m);
figure,imshow(m)
    imf(:,:,l) = h;
    I   =  I-h;
    l = l +1;
    figure,imshow(h)
    imwrite(h,'h.jpg')
end   
HE = histeq(I7);
CLAHE=adapthisteq(I);% CLAHE
figure,
subplot(1,2,1);
imshow(I7), title('original image');
subplot(1,2,2);
imhist(I7);title('Histogram')
figure,
subplot(1,2,1)
imshow(HE), title('HE');
subplot(1,2,2)
imhist(HE);title('Histogram')
figure,
subplot(1,2,1)
imshow(CLAHE), title('CLAHE');
subplot(1,2,2)
imhist(CLAHE);title('Histogram')
tic;
 addpath('selective_search');
if exist('selective_search/SelectiveSearchCodeIJCV')
  addpath('selective_search/SelectiveSearchCodeIJCV');
  addpath('selective_search/SelectiveSearchCodeIJCV/Dependencies');
else
  fprintf('Warning: you will need the selective search IJCV code.\n');
  fprintf('Press any key to download it (runs ./selective_search/fetch_selective_search.sh)> ');
  pause;
  system('./selective_search/fetch_selective_search.sh');
  addpath('selective_search/SelectiveSearchCodeIJCV');
  addpath('selective_search/SelectiveSearchCodeIJCV/Dependencies');
end
addpath('vis');
addpath('utils');
addpath('bin');
addpath('nms');
addpath('finetuning');
addpath('bbox_regression');
if exist('external/caffe/matlab/caffe')
  addpath('external/caffe/matlab/caffe');
else
  warning('Please install Caffe in ./external/caffe');
end
addpath('experiments');
addpath('imdb');
addpath('vis/pool5-explorer');
addpath('examples');
fprintf('R-CNN startup done\n');

% % %     done classification Using RCNN
     class = classify(convnet,I7);
     msgbox(char(class))
     
msgbox("Deep learning of Glaucoma classification  successfully completed by using R-CNN");
toc;


disp('%%%%----------RFCN CLASSIFICATION---------------%%%');


%%% extract the FCN features from the existed FCN structure
%%% the platform is based on the MatConvNet (CNNs for MATLAB)
%%% In our work, we extract the fine layer pool1(the 6th)
%%% and the coarse layer pool5(the 32th) 
%%% and save the cell results in .mat file named layer6 and layer32.
%%% and save it in FCN file.
%%% 2 add your image dataset path to 'imgFile' variable
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% add the model path
   addpath('matlab');
   addpath('FCN'); 
%      imgFile = imageDatastore('C:\Users\MEBIN\Desktop\fundusmathworks\Database',...
%     'IncludeSubfolders',true,... 
%     'LabelSource','foldernames');
  %% load the Net
  global res;
  imgFile = 'C:\Users\anush\Downloads\ExtractFCNFeature-master\ExtractFCNFeature-master\Extract_FCN_features\ACRIMA';    % the path of the image dataset
  outFile_6Level = './6/';           % the 6th layer
  outFile_32Level = './32/';         % the 32th layer
  imgPath = dir([imgFile  '*.bmp']); % you can modify the image format.
  for len = 1:length(imgPath)
      disp(len);                                             
      run matlab/vl_setupnn;                                               
      net = dagnn.DagNN.loadobj(load('FCN/pascal-fcn8s-dag.mat'));  
      net.mode = 'test';
      %% load and preprocess the image
    
      im = imread([imgFile imgPath(len).name] );
      im_= single(im);
      im_= imresize(im_,net.meta.normalization.imageSize(1:2));
      im_= bsxfun(@minus,im_ ,net.meta.normalization.averageImage);      
      %% run the CNN
      net.eval({'data',im_});
      layer6 = res{6};
      addr = [outFile_6Level imgPath(len).name(1:end-4) '.mat'];
      save(addr,'layer6');
      layer32 = res{32};
      addr = [outFile_32Level imgPath(len).name(1:end-4) '.mat'];
      save(addr,'layer32');     
  end
%%%%%
%Training by RFCN
options = trainingOptions('sgdm', ...
  'MiniBatchSize',32, ...
  'InitialLearnRate', 1e-6, ...
  'MaxEpochs',20);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% extract the FCN features from the existed FCN structure
%%% the platform is based on the MatConvNet (CNNs for MATLAB)
%%% In our work, we extract the fine layer pool1(the 6th)
%%% and the coarse layer pool5(the 32th) 
%%% and save the cell results in .mat file named layer6 and layer32.
%%% and save it in FCN file.
%%% 2 add your image dataset path to 'imgFile' variable
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% add the model path
tic;
   addpath('matlab');
   addpath('FCN'); 
%      imgFile = imageDatastore('C:\Users\MEBIN\Desktop\fundusmathworks\Database',...
%     'IncludeSubfolders',true,... 
%     'LabelSource','foldernames');
  %% load the Net
  global res;
  
  imgFile = 'C:\Users\anush\Downloads\ExtractFCNFeature-master\ExtractFCNFeature-master\Extract_FCN_features\ACRIMA';    % the path of the image dataset
  outFile_6Level = './6/';           % the 6th layer
  outFile_32Level = './32/';         % the 32th layer
  imgPath = dir([imgFile  '*.bmp']); % you can modify the image format.
  for len = 1:length(imgPath)
      disp(len);                                             
      run matlab/vl_setupnn;                                               
      net = dagnn.DagNN.loadobj(load('FCN/pascal-fcn8s-dag.mat'));  
      net.mode = 'test';
      %% load and preprocess the image
      im = imread([imgFile imgPath(len).name] );
      im_= single(im);
      im_= imresize(im_,net.meta.normalization.imageSize(1:2));
      im_= bsxfun(@minus,im_ ,net.meta.normalization.averageImage);      
      %% run the CNN
      net.eval({'data',im_});
      layer6 = res{6};
      addr = [outFile_6Level imgPath(len).name(1:end-4) '.mat'];
      save(addr,'layer6');
      layer32 = res{32};
      addr = [outFile_32Level imgPath(len).name(1:end-4) '.mat'];
      save(addr,'layer32');     
  end
% RFCN
 [convnet1, traininfo] = trainNetwork(trainData,lgraph,options);
 % % %     done classification Using RFCN
     class = classify(convnet1,I7);
     msgbox(char(class))
msgbox("Deep learning of Glaucoma classification  successfully completed by using RFCN");
toc;
