function [EEG, EOG] = Read_EEG_Set(subjectId, iArea, iB)
%LOAD_EEG Summary of this function goes here
%   e.g. Load_EEG(21);
%   dataEEG and dataEOG are cell arrays of shape iArea(3) x iB(2), and contain
%   double matrices of shape iFrame x iTrial
% TODO understand if it makes sense to keep b1 and b2 separate
% TODO check which areas are available or not

loadEOG = nargout > 1;
subjectStr = sprintf('%03d', subjectId);


%NOEMI changed new EP SSR. uncomment following if with old
%d = fullfile(get_dataset_dir(), 'EP_SSR_CRPP', filesep);
d = fullfile(get_dataset_dir(), 'EP_SSR', filesep);

global AreaNamesData AreaFullnames;

warning off MATLAB:table:ModifiedAndSavedVarnames;

area = AreaNamesData{iArea};
iBstr = num2str(iB);
text = ['S' subjectStr ', area ' area, ', b' iBstr];
pathEEG = [d 'CHEP/' AreaFullnames{iArea} '/1_Cz/chep_' area '_cz_b' iBstr '_crpp_' subjectStr '.txt'];
if exist(pathEEG, 'file')
    T = readtable(pathEEG, 'HeaderLines', 3);
    EEG = table2array(T)';
    EEG = EEG * 1e6;  % Convert to uV
    
    if loadEOG
        pathEOG = [d 'CHEP/' AreaFullnames{iArea} '/5_EOG/chep_' area '_eog_b' iBstr '_crpp_' subjectStr '.txt'];
        if exist(pathEOG, 'file')
            T = readtable(pathEOG, 'HeaderLines', 3);
            EOG = table2array(T)';
            EOG = EOG * 1e6;
            if ~all(size(EOG) == size(EEG))
                warning('EOG size for %s incompatible with EEG, dropped.', text);
                EOG = [];
            end
            % Why was this done?
            %dataEOG{iArea, iB}(20001:20002,:)=[];
            %dataEOG{iArea, iB}(:,size(dataEOG{iArea, iB},2)+1:size(dataEOG{iArea, iB},2))=[];
        else
            warning(['Missing EOG for ' text]);
            EOG = [];
        end
    end
else
    warning(['Missing EEG for ' text]);
    EEG = [];
    EOG = [];
end

end

