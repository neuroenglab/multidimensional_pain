function Trials = Denoise_EEG_from_EOG(Trials, loadDenoised)
global c last;

dataDir = load_data_dir();
eegDir = fullfile(dataDir, 'Analysis EEG', filesep);
denoisedDir = fullfile(eegDir, 'EOG denoised', filesep);

notMissing = ~cellfun(@isempty, Trials.EEG) & ~cellfun(@isempty, Trials.EOG);
[G, Gid] = findgroups(Trials.id);
for iG = 1:numel(Gid)
    idx = G == iG;
    subject = Gid{iG};
    
    rowsToCompute = idx & notMissing;
    
    n = sum(rowsToCompute);
    if n == 0
        warning('EOG unavailable for %s.', subject);
        continue;
    end
    
    keys = {'id', 'Area', 'B', 'iTrial'};
    p = sprintf('%sEEG_denoised_%s.mat', denoisedDir, subject);
    loaded = false;
    if loadDenoised && exist(p, 'file')
        fprintf('Loading denoised EEG for %s...\n', subject);
        load(p, 'SubjectEEG');
        try
            T = join(Trials(rowsToCompute, :), SubjectEEG, 'Keys', keys, 'LeftVariables', []);
            Trials.EEG(rowsToCompute) = T.EEG;
            loaded = true;
        catch
            disp('Incompatible denoised EEG, recomputing...');
        end
    end
    if ~loaded
        c = 0;
        last = -1;
        fprintf('Denoising EEG from EOG of %s:     ', subject);
        EEG = rowfun(@(eeg,eog) filter_trial(eeg,eog,n), Trials(rowsToCompute, {'EEG','EOG'}), 'OutputFormat', 'cell');
        fprintf('\n');
        Trials.EEG(rowsToCompute) = EEG;
        SubjectEEG = Trials(rowsToCompute, [keys {'EEG'}]);
        save(mkdir_file(p), 'SubjectEEG');
    end
end

end

function trialEEGfilt = filter_trial(trialEEG, trialEOG, n)
global c last;
c = c + 1;
progress = round(c/n*100);
if progress > last
    fprintf('\b\b\b\b%3d%%', progress);
    last = progress;
end

trialEEG = trialEEG{1};
trialEOG = trialEOG{1};
trialEEG = trialEEG - mean(trialEEG);
trialEOG = trialEOG - mean(trialEOG);

fs = 2000;
fc = 16;
%[b,a] = butter(3,fc/(fs/2));
%freqz(b,a);
%trialEOG = filter(b,a,trialEOG);
%trialEOG = lowpass(trialEOG,fc,fs,'Steepness',0.99);

l = 64;
lms = dsp.LMSFilter('Method', 'Normalized LMS', 'Length', l);
[y,err] = lms(trialEOG',trialEEG');
% err = trialEEG' - y == denoised EEG
% y(t) = wts x trialEOG(t-1)
plot_all = false;
if plot_all
    figure;
    hold on;
    plot(trialEEG);
    plot(trialEOG*0.5+200);
    figure;
    tiledlayout('flow');
    nexttile;
    plot(trialEEG);
    nexttile;
    plot(trialEOG);
    nexttile;
    plot(y);
    nexttile;
    plot(err);
    figure;
    hold on;
    yyaxis left;
    plot(err);  % This is the denoised EEG
    yyaxis right;
    plot(trialEOG);
end

% Low-pass filter correction term
yFilt = lowpass(y,fc,fs,'Steepness',0.999);
trialEEGfilt = (trialEEG'-yFilt)';

end
