init;
dataDir = load_data_dir();

reloadData = false;  % If true, load previously preprocessed data
PreprocessedData_path='PreprocessedData.mat';

%% Preprocess Data
preprocDataPath = fullfile(dataDir, PreprocessedData_path);
if isfile(preprocDataPath) && reloadData
    disp('Reloading preprocessed data...');
    load(preprocDataPath, 'Subjects', 'SubjectAreas', 'Trials');
else
    % Load Raw Data
    rawDataPath = fullfile(dataDir, 'RawData.mat');
    if isfile(rawDataPath) && reloadData
        disp('Reloading raw data...');
        load(rawDataPath, 'Subjects', 'SubjectAreas', 'RawTrials');
    else
        [Subjects, SubjectAreas, RawTrials] = Read_Raw_Data();
        save(rawDataPath, 'Subjects', 'SubjectAreas', 'RawTrials', '-v7.3');
    end
    [Subjects, SubjectAreas, Trials] = Preprocess_Data(Subjects, SubjectAreas, RawTrials);
    Trials=processing_SC_peaks(Trials);
    save(preprocDataPath, 'Subjects', 'SubjectAreas', 'Trials', '-v7.3');
end

save_features_path='Trials.csv';

%% Compute features
fs = 2000;
scBaselineStart = 13975/fs;
scBaselineEnd = 18975/fs;
scPainStart = 4000/fs;
scPainEnd = 9000/fs;
eegBaselineStart = 16975/fs;
eegBaselineEnd = 18975/fs;
eegPainStart = 1000/fs;
eegPainEnd = 3000/fs;
   
compute_features(Trials, scBaselineStart, scBaselineEnd, scPainStart, scPainEnd, eegBaselineStart, eegBaselineEnd, eegPainStart, eegPainEnd, fs, save_features_path)

function compute_features(Trials, scBaselineStart, scBaselineEnd, scPainStart, scPainEnd, eegBaselineStart, eegBaselineEnd, eegPainStart, eegPainEnd, fs, save_features_path)
    [TrialsBinary, splitName] = split_trials_binary(Trials, ...
        scBaselineStart, scBaselineEnd, scPainStart, scPainEnd, eegBaselineStart, eegBaselineEnd, eegPainStart, eegPainEnd, fs);

    outputFolder = fullfile(load_data_dir(), 'Baseline-Pain data', splitName, filesep);
    if ~isfolder(outputFolder)
        mkdir(outputFolder)
    end

    % Compute SC features
    disp('Computing SC features...');
    TrialsBinary = Compute_SC_Features(TrialsBinary, false);

    % Compute EEG features
    disp('Computing EEG features...');
    TrialsBinary = Compute_EEG_Features(TrialsBinary, false);

    disp(head(TrialsBinary));
    summary(TrialsBinary)

    % Drop timeseries
    timeseries = {'SCH','SCF','EEG','EOG','SCH_1D','SCF_1D','SCH_2D','SCF_2D'};
    TrialsFeatures = removevars(TrialsBinary, timeseries);

    writetable(TrialsFeatures, fullfile(outputFolder, save_features_path),'Delimiter', ',')
end
