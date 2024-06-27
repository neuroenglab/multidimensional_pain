function Trials = split_timeseries(Trials, columnName, baselineStart,baselineEnd, painStart, painEnd)
missing = cellfun(@isempty, Trials.(columnName));
blColumnName = [columnName '_BL'];
Trials.(blColumnName)(~missing) = cellfun(@(c) c(baselineStart:baselineEnd), Trials.(columnName)(~missing), 'uni', false);
% could put into a new column (e.g. _post) instead
Trials.(columnName)(~missing) = cellfun(@(c) c(painStart:painEnd), Trials.(columnName)(~missing), 'uni', false);
end

