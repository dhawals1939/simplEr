function [searchN, V, data] = computePeakPhotonsPerVoltage(staticCmd, spatialX, spatialY, n_max_min, n_max_max, nSearches, fx, fy)
          
if(nargin <= 6)
    fx = (spatialX+1)/2;
    fy = (spatialX+1)/2;
end
searchN = linspace(n_max_min, n_max_max, nSearches);
data = zeros(spatialX, spatialY, nSearches);

for i=1:length(searchN)
    cmd = staticCmd + ...       
            "spatialX=" + num2str(spatialX) + " " + ...
            "spatialY=" + num2str(spatialY) + " " + ...
            "outFilePrefix=delete " + ...
            "n_max=" + num2str(searchN(i));
    system(cmd);
    temp = readpfm3d('delete.pfm3d');
    data(:, :, i) = temp;
end
if(fx == round(fx) && fy == round(fy))
    V = squeeze( data(fx, fy, :) );
else
    V = squeeze( data(ceil(fx), ceil(fy), :)   * abs(ceil(fx) - fx)  * abs(ceil(fx) - fy)  + ... 
                 data(floor(fx), ceil(fy), :)  * abs(floor(fx) - fx) * abs(ceil(fx) - fy)  + ... 
                 data(ceil(fx), floor(fy), :)  * abs(ceil(fx) - fx)  * abs(floor(fx) - fy) + ... 
                 data(floor(fx), floor(fy), :) * abs(floor(fx) - fx) * abs(floor(fx) - fy)   ... 
            )/(abs(ceil(fx) - fx)*abs(ceil(fx) - fy) + abs(floor(fx) - fx)*abs(ceil(fx) - fy) + abs(ceil(fx) - fx)*abs(floor(fx) - fy) + abs(floor(fx) - fx)*abs(floor(fx) - fy));
end