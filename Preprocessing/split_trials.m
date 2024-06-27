function [Trials, splitName] = split_trials(Trials, scBaselineStart, scBaselineEnd, scPainStart, scPainEnd, eegBaselineStart, eegBaselineEnd, eegPainStart, eegPainEnd, fs)
% Inputs are in seconds
timestamps = [scBaselineStart, scBaselineEnd, scPainStart, scPainEnd, ...
              eegBaselineStart, eegBaselineEnd, eegPainStart, eegPainEnd];
splitName = sprintf('SC_BL %g-%g SC %g-%g EEG_BL %g-%g EEG %g-%g', timestamps);

timestamps = round(timestamps * fs);
timestamps(1:2:end) = timestamps(1:2:end) + 1;

Trials = split_timeseries(Trials, 'SCH', timestamps(1), timestamps(2), timestamps(3), timestamps(4));
Trials = split_timeseries(Trials, 'SCF', timestamps(1), timestamps(2), timestamps(3), timestamps(4));
Trials = split_timeseries(Trials, 'EEG', timestamps(5), timestamps(6), timestamps(7), timestamps(8));

end

