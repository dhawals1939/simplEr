function [searchN, V] = computePeakPhotonsPerVoltage(staticCmd, spatialX, spatialY, n_max_min, n_max_max, nSearches)
          
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
V = squeeze(data(ceil(end/2), ceil(end/2), :));