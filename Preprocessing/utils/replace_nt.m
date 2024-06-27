function Table = replace_nt(Table)

nt = cellfun(@(c) strcmp(c,'nt'), table2cell(Table));
[iNT, jNT] = find(nt);
for i = 1:numel(iNT)
    Table{iNT(i), jNT(i)} = {''};
end

end

