function dataDir = load_data_dir()
if exist('../dataDir.txt','file')
    fid = fopen('../dataDir.txt');
    dataDir = fgetl(fid);
    fclose(fid);
else
    dataDir = set_data_dir();
end
end

