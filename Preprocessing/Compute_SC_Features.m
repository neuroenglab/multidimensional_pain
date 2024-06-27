function Trials = Compute_SC_Features(Trials, hasBL)

if ~exist('hasBL', 'var')
    hasBL = true;
end

baseColumnNames = {'SCH', 'SCF'};
if hasBL
    blColumnNames = {'SCH_BL', 'SCF_BL'};
    columnNames = [baseColumnNames blColumnNames];
else
    columnNames = baseColumnNames;
end

%COMPUTE DERIVATIVE
Trials.SCH_1D=cellfun(@gradient, Trials.SCH, 'UniformOutput', false);
Trials.SCF_1D=cellfun(@gradient, Trials.SCF, 'UniformOutput', false);
Trials.SCH_2D=cellfun(@gradient, Trials.SCH_1D, 'UniformOutput', false);
Trials.SCF_2D=cellfun(@gradient, Trials.SCF_1D, 'UniformOutput', false);
derivative_columnNames=["SCH_1D","SCH_2D","SCF_1D","SCF_2D"];

%SIGNAL FEATURES
Trials = compute_feature(@(c) abs(mean(c)), 'absmean', columnNames, Trials);
Trials = compute_feature(@rms, 'rms', columnNames, Trials);
Trials = compute_feature(@mad, 'mad', columnNames, Trials);
%Trials = compute_feature(@auc, 'auc', columnNames, Trials);
Trials = compute_feature(@(c) max(c,[],'all') - min(c,[],'all'), 'range', columnNames, Trials);
Trials = compute_feature(@iqr, 'iqr', columnNames, Trials);
Trials = compute_feature(@var, 'var', columnNames, Trials);

%DERIVATIVE FEATURES
Trials= compute_feature(@var, 'var', derivative_columnNames, Trials);
Trials= compute_feature(@mean, 'mean', derivative_columnNames, Trials);
Trials= compute_feature(@max, 'max', derivative_columnNames, Trials);
Trials= compute_feature(@min, 'min', derivative_columnNames, Trials);

% std can be calculated faster from var column, and can overlap with rms
%Trials = compute_feature(@std, 'std', columnNames, Trials);
%Trials = compute_feature(@median, 'med', columnNames, Trials);
%zcd = dsp.ZeroCrossingDetector;


for i = 1:numel(baseColumnNames)
    c = baseColumnNames{i};
    [SampEn_01, SampEn_02, SampEn_01_eu]=cellfun(@compute_SampEn, Trials.(c));
    Trials.([c '_SampEn_01'])=SampEn_01;
    Trials.([c '_SampEn_02'])=SampEn_02;
    Trials.([c 'SampEn_01_eu'])=SampEn_01_eu;
    
    [PeakAmplitude, PeakTime, DeltaAmplitude, DeltaTime,NPeaks] = cellfun(@find_peaks_sc, Trials.(c));
    Trials.([c '_PeakAmplitude']) = PeakAmplitude;
    Trials.([c '_PeakTime']) = PeakTime;
    Trials.([c '_DeltaAmplitude']) = DeltaAmplitude;
    Trials.([c '_DeltaTime']) = DeltaTime;
    Trials.([c '_NPeaks']) = NPeaks;
    Trials.([c '_ZC'])=cellfun(@compute_ZC, Trials.(c));
    %Trials.([c '_ZC']) = cellfun(@zcd, Trials.(c));
end

if hasBL
    %columnsToDiff = {'auc','range','var'};
    columnsToDiff = {'absmean', 'rms', 'mad', 'range', 'iqr', 'var'};
    Trials = compute_diff(Trials, baseColumnNames, columnsToDiff);
end
end

function [nZC]=compute_ZC(Signal)
    zci = @(v) find(v(:).*circshift(v(:), [-1 0]) <= 0);
    ZC=zci(Signal);
    ZC=ZC(ZC~=length(Signal));
    if isempty(ZC)
        nZC=0;
    else
        nZC=length(ZC);
    end 
end

function [SampEn_01, SampEn_02, SampEn_01_eu]=compute_SampEn(Signal)
    if isempty(Signal)
        SampEn_01 = NaN;
        SampEn_02 = NaN;
        SampEn_01_eu=NaN;
        return;
    end
    SampEn_01 = sampen(Signal,2,0.1,'chebychev');
    SampEn_02 = sampen(Signal,2,0.2,'chebychev');
    SampEn_01_eu = sampen(Signal,2,0.1,'euclidean');
end

function [PeakAmplitude, PeakTime, DeltaAmplitude, DeltaTime, NPeaks] = find_peaks_sc(Signal)
    %   If there is not signals --> nan 
    if isempty(Signal)
        PeakAmplitude = NaN;
        PeakTime = NaN;
        DeltaAmplitude=NaN;
        DeltaTime= NaN;
        NPeaks=NaN;
        return;
    end
    
    %   Compute peaks and indexes (positive and negative)
    [PeaksP, MaxH_IdsP] = findpeaks(Signal);
    
    % if the algorithm cannot find peaks --> nan
    if (isempty(PeaksP))
        PeakAmplitude = NaN;
        PeakTime = NaN;
        DeltaAmplitude=NaN;
        DeltaTime= NaN;
        NPeaks=NaN;
    %else find the maximum and the delta and relative latencies
    else
        [PeakAmplitude, PeakTime, DeltaAmplitude, DeltaTime]=peaks_util(PeaksP, MaxH_IdsP, Signal);
        NPeaks=length(PeaksP);
    end   
end


function [PeakAmplitude, PeakTime, DeltaAmplitude,DeltaTime]=peaks_util(Peaks, MaxH_Ids, Signal)
% find maximum peak
[PeakAmplitude, ID_Peak] = max(Peaks);
PeakTime = MaxH_Ids(ID_Peak);

% compute delta time and delta amplitude from the onrise of the peak
TF = islocalmin(Signal);
LocMinI = find(TF);
[~,MinI]= min(Signal);
AllMinI = [LocMinI, MinI];
Distance = PeakTime - AllMinI;
[DeltaTime,~] = min(Distance(Distance>0));
if isempty(find(Distance>0, 1))
    DeltaTime = PeakTime-1;
end
DeltaAmplitude = PeakAmplitude-Signal(PeakTime-DeltaTime);
end

