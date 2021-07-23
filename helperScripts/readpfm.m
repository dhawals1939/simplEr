function D = readpfm(filename_pfm)

fid = fopen(filename_pfm);

if(fid < 0)
    warning(strcat(filename_pfm," does not exist"));
    return;
end

fscanf(fid,'%c',[1,3]);
cols = fscanf(fid,'%f',1);
rows = fscanf(fid,'%f',1);
fscanf(fid,'%f',1);
fscanf(fid,'%c',1);
D = fread(fid,[cols,rows],'double');
D(D == Inf) = 0;
fclose(fid);