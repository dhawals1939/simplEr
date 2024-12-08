function D = readpfm3d(filename_pfm, precision)

if(nargin == 1)
  precision = 'double';
end

fid = fopen(filename_pfm);

if(fid < 0)
    warning(strcat(filename_pfm," does not exist"));
    return;
end

fscanf(fid,'%c',[1,3]);
cols = fscanf(fid,'%f',1);
rows = fscanf(fid,'%f',1);
depths = fscanf(fid,'%f',1);
fscanf(fid,'%f',1);
fscanf(fid,'%c',1);
D = zeros(rows, cols, depths);

for i=1:depths
    D(:,:,i) = fread(fid,[cols,rows],precision)';
end

D(D == Inf) = 0;
fclose(fid);
