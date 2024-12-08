clc, clear, close all

dim = 1000;
slopeAngle = 5;

I = zeros(dim, dim);

dimy = 1:dim;

dimx = (dim/2-dimy)* tand(slopeAngle) + dim/2;

for i=1:length(dimy)
    I(dimy(i), round(dimx(i)):end)=1;
end

imshow(I);

pfmwrite(I, "slantEdge.pfm");