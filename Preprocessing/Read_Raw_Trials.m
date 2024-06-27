function RawTrials = Read_Raw_Trials(Subjects)
%% Load SC
disp('Reading SC data...');
TrialsSC = Read_SC(Subjects);

%% Load NRS
disp('Reading NRS...');
TrialsSC = Read_NRS(TrialsSC);

%% Load EEG
disp('Reading EEG data...');
RawTrials = Read_EEG(TrialsSC);

end

