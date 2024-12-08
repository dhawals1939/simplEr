function [n, searchN, V, smoothV, data]= computeFocusVoltage(staticCmd, spatialX, spatialY, n_max_min, n_max_max, nSearches, gVar, fx, fy)

if(nargin <= 8)
    [searchN, V, data] = computePeakPhotonsPerVoltage(staticCmd, spatialX, spatialY, n_max_min, n_max_max, nSearches);
else
    [searchN, V, data] = computePeakPhotonsPerVoltage(staticCmd, spatialX, spatialY, n_max_min, n_max_max, nSearches, fx, fy);
end
smoothV = smoothdata(V,'gaussian',gVar);
[pks, locs] = findpeaks(smoothV);

if length(pks) > 1
    disp('More than one peak found. Focus voltage might be inaccurate');
end
n = searchN(locs(1));