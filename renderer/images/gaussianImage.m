clc, clear, close all

FWHM = 100; % 100 pixels
sigma = FWHM/(2*sqrt(2*log(2))); 
dim = 6*round(sigma)+1;

G = fspecial('gaussian', [dim, dim], sigma);

imshow(G,[]);

pfmwrite(G, "GaussianFilterFWHM100um_6sigma253um.pfm");

