function [n, searchN, V, smoothV]= computeFocusVoltage(staticCmd, spatialX, spatialY, n_max_min, n_max_max, nSearches, gVar)

[searchN, V] = computePeakPhotonsPerVoltage(staticCmd, spatialX, spatialY, n_max_min, n_max_max, nSearches);
smoothV = smoothdata(V,'gaussian',gVar);
[pks, locs] = findpeaks(smoothV);

if length(pks) > 1
    disp('More than one peak found. Focus voltage might be inaccurate');
end
n = searchN(locs(1));