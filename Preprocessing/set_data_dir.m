function dataDir = set_data_dir()
dataDir = [uigetdir(pwd, 'Select Data directory') filesep];
fid = fopen('../dataDir.txt','wt');
fprintf(fid, '%s', dataDir);
fclose(fid);
end

