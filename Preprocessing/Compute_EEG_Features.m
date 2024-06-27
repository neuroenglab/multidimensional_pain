function Trials = Compute_EEG_Features(Trials, hasBL)

if ~exist('hasBL', 'var')
    hasBL = true;
end

if hasBL
    columnNames = {'EEG', 'EEG_BL'};
else
    columnNames = {'EEG'};
end
Trials = compute_feature(@mean, 'mean', columnNames, Trials);
Trials = compute_feature(@rms, 'rms', columnNames, Trials);
% Trials = compute_feature(@mad, 'mad', columnNames, Trials);
% Trials = compute_feature(@(c) trapz(c-min(c)), 'auc', columnNames, Trials);
% Trials = compute_feature(@(c) max(c) - min(c), 'range', columnNames, Trials);
% Trials = compute_feature(@iqr, 'iqr', columnNames, Trials);
% Trials = compute_feature(@std, 'std', columnNames, Trials);
% Trials = compute_feature(@var, 'var', columnNames, Trials);
% Trials = compute_feature(@median, 'med', columnNames, Trials);
Trials = compute_feature(@skewness, 'skew', columnNames, Trials); % new implementation chia
% Trials = compute_feature(@kurtosis, 'kurtosis', columnNames, Trials); % new implementation chia: biased
Trials = compute_feature(@(x) kurtosis(x, 0), 'kurtosis', columnNames, Trials); % new implementation chia - unbiased


for i = 1:numel(columnNames)
    c = columnNames{i};
    [delta, theta,alpha, beta,  gamma,total_auc_frequency, peak_frequency, peak_alpha_frequency, peak_beta_frequency] = cellfun(@find_frequency_bands, Trials.(c));
    Trials.([c '_delta'])=delta;
    Trials.([c '_alpha'])=alpha;
    Trials.([c '_beta'])=beta;
    Trials.([c '_theta'])=theta;
    Trials.([c '_gamma'])=gamma;
    Trials.([c '_total_auc_frequency'])=total_auc_frequency;
    Trials.([c '_peak_frequency'])=peak_frequency;
    Trials.([c '_peak_alpha_frequency'])=peak_alpha_frequency;
    Trials.([c '_peak_beta_frequency'])=peak_beta_frequency;

    fprintf("computing EEG sample entropy\n")
    % sample entropy
    [SampEn_01, SampEn_02, SampEn_01_m4]=cellfun(@compute_SampEn, Trials.(c));
    Trials.([c '_SampEn_01'])=SampEn_01;
    Trials.([c '_SampEn_02'])=SampEn_02;
    Trials.([c 'SampEn_01_m4'])=SampEn_01_m4;
end

if hasBL
    columnsToDiff = {'mean','rms','delta','alpha','beta','theta','gamma','total_auc_frequency'};
    Trials = compute_diff(Trials, {'EEG'}, columnsToDiff);
end
end


function [delta, theta,alpha, beta,  gamma, total_auc_frequency, peak_frequency, peak_alpha_frequency, peak_beta_frequency]=find_frequency_bands(Signal)
window=1000;
noverlap=750;
nfft=2000;
fs=2000;
[pxx, f]=pwelch(Signal, window, noverlap, nfft, fs);
%[pxxp,fp] = periodogram(Signal,[],length(Signal),fs) ;
%figure; plot(f, pxx); hold on; area(f(f>=4 & f<=7.5),pxx(f>=4 & f<=7.5)); area(f(f>7.5 & f<=13),pxx(f>7.5 & f<=13)); area(f(f>13 & f<=29),pxx(f>13 & f<=29))
total_auc_frequency=trapz(f, pxx);
delta=trapz(f(f<4),pxx(f<4));
alpha=trapz(f(f>=8 & f<=13),pxx(f>=8 & f<=13));
theta=trapz(f(f>=4 & f<8),pxx(f>=4 & f<8));
gamma=trapz(f(f>29 & f<=45),pxx(f>29 & f<=45));
beta=trapz(f(f>13 & f<=29),pxx(f>13 & f<=29));



% peak frequency from chaira
range_peak_frequency = (f>=4 & f<=13);
f_range = f(range_peak_frequency);
peak_frequency = f_range(pxx(range_peak_frequency) == max(pxx(range_peak_frequency)));

% peak alpha frequency - between 8 and 12 Hz
alpha_frequency = f(f>=8 & f<=13);
alpha_band = pxx(f>=8 & f<=13);
peak_alpha_frequency = alpha_frequency(alpha_band == max(alpha_band));

% peak beta frequency
beta_frequency = f(f>13 & f<=29);
beta_band = pxx(f>13 & f<=29);
peak_beta_frequency = beta_frequency(beta_band == max(beta_band));

% 
% % peak frequency and alpha peak
% peak_frequency = f(pxx == max(pxx));
% peak_alpha_frequency = alpha_frequency(alpha_band == max(alpha_band));

end

function [SampEn_01, SampEn_02, SampEn_01_m4] = compute_SampEn(Signal)
    if isempty(Signal)
        SampEn_01 = NaN;
        SampEn_02 = NaN;
        SampEn_01_m4 = NaN;
        return;
    end
    SampEn_01 = sampen(Signal,2,0.1,'chebychev');
    SampEn_02 = sampen(Signal,2,0.2,'chebychev');
    SampEn_01_m4 = sampen(Signal,4,0.1,'chebychev');
end

