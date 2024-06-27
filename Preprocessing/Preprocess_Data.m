function [Subjects, SubjectAreas, Trials] = Preprocess_Data(Subjects, SubjectAreas, RawTrials, cut)
if ~exist('cut', 'var')
    cut = true;
end

% S025 chep_mp_ssrh_b1_crpp_025.txt (failed SSR recording)

ExcludedSubjects = 25;
ExcludedSubjectsIds = Subjects.id(ExcludedSubjects);
sExcl = join(ExcludedSubjectsIds, ', ');
RawTrials(ismember(RawTrials.id, ExcludedSubjectsIds), :) = [];
SubjectAreas(ismember(SubjectAreas.id, ExcludedSubjectsIds), :) = [];
Subjects(ismember(Subjects.Subject, ExcludedSubjects), :) = [];
sLeft = join(Subjects.id, ', ');
fprintf('Excluded subjects:\n%s.\nRemaining:\n%s.\n', sExcl{1}, sLeft{1});
Subjects = removevars(Subjects, 'Subject');

%% Load Raw Data
Trials = RawTrials;

%% Filter SC
disp('Filter SC...');
Trials = compute_feature(@Filter_SC, [], {'SCH', 'SCF'}, Trials, false);

%% Filter EEG CHANGE HERE FOR FILTERING EEG
disp('Filter EEG...');
Trials = Filter_EEG(Trials, true, 'bandpass');
i = cellfun(@numel, RawTrials.EEG) < 20000;
if cut
    Trials.EEG(~i) = cellfun(@(c) c(1001:end), Trials.EEG(~i), 'UniformOutput', false);
    Trials.SCH(~i) = cellfun(@(c) c(1001:end), Trials.SCH(~i), 'UniformOutput', false);
    Trials.SCF(~i) = cellfun(@(c) c(1001:end), Trials.SCF(~i), 'UniformOutput', false);
else 
    Trials(i,:)=[]; 
end

end

