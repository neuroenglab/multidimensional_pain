function [TrialsBinary, splitName] = split_trials_binary(Trials, ...
    scBaselineStart, scBaselineEnd, scPainStart, scPainEnd, eegBaselineStart, eegBaselineEnd, eegPainStart, eegPainEnd, fs)

TrialsPost = Trials;
TrialsBL = Trials;
TrialsPost.BL(:) = false;
TrialsBL.BL(:) = true;

[TrialsSplit, splitName] = split_trials(Trials, scBaselineStart, scBaselineEnd, scPainStart, scPainEnd, ...
                                   eegBaselineStart, eegBaselineEnd, eegPainStart, eegPainEnd, fs);

TrialsPost.SCH = TrialsSplit.SCH;
TrialsPost.SCF = TrialsSplit.SCF;
TrialsPost.EEG = TrialsSplit.EEG;

TrialsBL.SCH = TrialsSplit.SCH_BL;
TrialsBL.SCF = TrialsSplit.SCF_BL;
TrialsBL.EEG = TrialsSplit.EEG_BL;

TrialsBinary = [TrialsBL; TrialsPost];

end

