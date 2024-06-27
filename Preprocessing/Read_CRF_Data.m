function Table = Read_CRF_Data(sheet)
Table = readtable(get_crf_sheet_path(),'Sheet', sheet, 'DataRange', 'A2', 'VariableNamesRange', 'A1');
Table = replace_nt(Table);
end

