function Trials = Read_NRS(Trials)
global AreaNamesData;

for area = 1:numel(AreaNamesData)
    sheet = ['CHEP ' AreaNamesData{area}];
    % Can't use Read_CRF_Data because of custom import options
    Table = readtable(get_crf_sheet_path(),'Sheet', sheet, 'Format', 'auto', 'TreatAsEmpty', {'','na'});
    
    B1cols = startsWith(Table.Properties.VariableNames,'chep_b1_nrs_');
    B2cols = startsWith(Table.Properties.VariableNames,'chep_b2_nrs_');
    stringColumns = (B1cols | B2cols) & varfun(@iscell, Table, 'OutputFormat', 'uniform');
    stringColumnsB1 = find(B1cols & stringColumns);
    stringColumnsB2 = find(B2cols & stringColumns);
    
    B1_size = zeros(1, height(Table)) + sum(B1cols);
    B2_size = zeros(1, height(Table)) + sum(B2cols);
    for subject = 1:height(Table)
        iNT = find(strcmp(Table{subject, B1cols & stringColumns}, 'nt'), 1);
        if ~isempty(iNT)
            B1_size(subject) = sum(B1cols(1:stringColumnsB1(iNT)))-1;
        end
        iNT = find(strcmp(Table{subject, B2cols & stringColumns}, 'nt'), 1);
        if ~isempty(iNT)
            B2_size(subject) = sum(B2cols(1:stringColumnsB2(iNT)))-1;
        end
    end
    
    for c = find(stringColumns)
        % A bit ugly
        Table.(Table.Properties.VariableNames{c}) = str2double(Table{:, c});
    end
    
    for subject = 1:height(Table)
        NRSb1 = Table{subject, B1cols};
        NRSb2 = Table{subject, B2cols};
        NRSb1 = NRSb1(1:B1_size(subject));
        NRSb2 = NRSb2(1:B2_size(subject));
        subjTrials = strcmpi(Table.id(subject), Trials.id) & Trials.Area == area;
        if ~any(subjTrials)
            % Meaning that the subject has been excluded earlier
            continue;
        end
        B1_SC = subjTrials & Trials.B == 1;
        B2_SC = subjTrials & Trials.B == 2;
        if B1_size(subject) > sum(B1_SC)
            NRSb1 = NRSb1(1:sum(B1_SC));
        elseif B1_size(subject) < sum(B1_SC)
            NRSb1 = [NRSb1 nan(1, sum(B1_SC) - B1_size(subject))];
        end
        if B2_size(subject) > sum(B2_SC)
            NRSb2 = NRSb2(1:sum(B2_SC));
        elseif B2_size(subject) < sum(B2_SC)
            NRSb2 = [NRSb2 nan(1, sum(B2_SC) - B2_size(subject))];
        end
        Trials.NRS(subjTrials & Trials.B == 1) = NRSb1;
        Trials.NRS(subjTrials & Trials.B == 2) = NRSb2;
    end
end
end

