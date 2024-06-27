function Trials = compute_diff(Trials, baseColumnNames, columnsToDiff)
% Calculates relative difference between Post and BL

for i = 1:numel(baseColumnNames)
    cBase = baseColumnNames{i};
    cBL = [cBase '_BL'];
    for iVar = 1:numel(columnsToDiff)
        cPre = [cBL '_' columnsToDiff{iVar}];
        cPost = [cBase '_' columnsToDiff{iVar}];
        cDiff = [cPost '_diff'];
        Trials.(cDiff) = (Trials.(cPost)-Trials.(cPre))./Trials.(cPre);
        Trials = movevars(Trials, cDiff, 'After', cPost);
    end
end

end

