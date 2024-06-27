function Trials = processing_SC_peaks(Trials)
baseColumnNames = {'SCH', 'SCF'};
for i = 1:numel(baseColumnNames)
    c = baseColumnNames{i};

    Trials.(c) = cellfun(@find_maximum, Trials.(c), 'UniformOutput', false);
end
end


function Signal_new = find_maximum(Signal)
    if isempty(Signal)
       Signal_new=[];
       return;
    end
    [PeaksP, MaxH_IdsP] = findpeaks(Signal);
    [mP,iP]=max(PeaksP);
    [PeaksN, ~] = findpeaks(-Signal);
    [mN, ~] = max(PeaksN);
    if ((isempty(mN))&&(isempty(mP)))
        Signal_new=Signal;
    elseif ((~isempty(mN))&&(isempty(mP)))
        Signal_new=-Signal;
    elseif ((isempty(mN))&&(~isempty(mP)))
        Signal_new=Signal;
    elseif (mP<mN)&&(MaxH_IdsP(iP)>10000)
        Signal_new=-Signal; 
    else 
        Signal_new=Signal;
    end
end


