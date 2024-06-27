function Trials = compute_feature(func, funcName, columnNames, Trials, scalarOutput)
% scalarOutput is optional (default true)
% if funcName is empty, the column is replaced

if ~exist('scalarOutput','var')
    scalarOutput = true;
end

if ischar(columnNames)
    % If it is only one columm
    columnNames = {columnNames};
end
for i = 1:numel(columnNames)
    c = columnNames{i};
    notMissing = ~cellfun(@isempty, Trials.(c));
    if isempty(funcName)
        newColumnName = c;
    else
        newColumnName = [c '_' funcName];
    end
    Trials.(newColumnName)(notMissing) = cellfun(func, Trials.(c)(notMissing), 'UniformOutput', scalarOutput);
    Trials = movevars(Trials, newColumnName, 'After', c);
end
end

