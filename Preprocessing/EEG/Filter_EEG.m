function Trials = Filter_EEG(Trials, filterEOG, freqFilter)
% freqFilter can be 'none', '50', or 'bandpass'

if filterEOG
    loadDenoised = true;
    Trials = Denoise_EEG_from_EOG(Trials, loadDenoised);
end

fs = 2000;
switch freqFilter
    case 'none'
    case '50'
        disp('Stopband filter 49.5 - 50.5 Hz');
        [b, a] = butter(2,[49.5 50.5]/(fs/2),'stop');
        Trials = compute_feature(@(x) filter(b, a, x), [], 'EEG', Trials, false);
    case 'bandpass'
        disp('Bandpass filter 0.1 - 45 Hz');
        [b,a]=butter(2,[0.1 45]/(fs/2), 'bandpass');
        Trials = compute_feature(@(x) filter(b, a, x), [], 'EEG', Trials, false);
    case 'FIR'
        order = 2000;
        f = fir1(order, [0.1 45]/(fs/2), 'BANDPASS', hamming(order+1));
        Trials = compute_feature(@(x) filter(f,1,x), [], 'EEG', Trials, false);
    otherwise
        error('Invalid freqFilter argument.');
end

end

