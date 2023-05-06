clear all;
close all;
clc;
% Read image
[filename,pathname]=uigetfile('*.jpeg','*.png','*.jpg');
Im1= imread(filename);
I = im2double(rgb2gray(Im1));
figure,imshow(I);title('Input Grey Image')
% Rotating Image
I = imrotate(I, 45);
figure,imshow(I);title('Input Rotate Image')
% Noise Removal
I1 = imnoise(rgb2gray(Im1),'salt & pepper',0.02);% CLIP LIMIT
subplot(1,2,1),imshow(I1);
title('Noise addition');
K = medfilt2(I1);
subplot(1,2,2),imshow(K);
title('Noise removal using median filter');
tic;
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
%% Empirical mode decomposition
EMD=emd('1.tif');
figure,imshow(EMD)
toc;

