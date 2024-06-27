function SubjectAreas = Read_QST(Subjects, SubjectAreas)
%{

Compute and store in a pre-existing structure the different pain threshold 
determined with the data of the QST experiment.

Input: - UserPath: Path to access the Balgrist Dataset folder
       - Dataset: Pre-existing structure containing subjective information for every subject

Output: - Dataset: Pre-existing structure containing now for every subject
        HPT, PPT and MPT results of QST experiment

%}

% References values for normalization based on QST paper

%ORDER F <40, F>=40, M<40, M>=40
Ref_Foot_HPT_mean = [43.81, 45.02, 45.14, 46.98];
Ref_Hand_HPT_mean = [42.61, 44.06, 44.14, 45.81];
Ref_Foot_HPT_std = [2.80, 2.68, 2.37, 2.14];
Ref_Hand_HPT_std = [3.33, 3.34, 2.77, 2.92];

Ref_Back_HPT_mean = [43.69, 44.31, 42.66, 45.63];
Ref_Back_HPT_std = [3.33, 3.69, 3.14, 3.24];

Ref_Foot_PPT_mean = [2.678, 2.728, 2.763, 2.756];
Ref_Hand_PPT_mean = [2.544, 2.644, 2.627, 2.713];
Ref_Foot_PPT_std = [0.118, 0.147, 0.183, 0.164];
Ref_Hand_PPT_std = [0.108, 0.115, 0.173, 0.134];

Ref_Back_PPT_mean = [2.590, 2.620, 2.741, 2.788];
Ref_Back_PPT_std = [0.120, 0.151, 0.114, 0.160];

Ref_Foot_MPT_mean = [1.831, 1.796, 1.867, 2.014];
Ref_Hand_MPT_mean = [1.889, 1.852, 1.912, 2.066];
Ref_Foot_MPT_std = [0.410, 0.344, 0.409, 0.392];
Ref_Hand_MPT_std = [0.348, 0.342, 0.431, 0.334];

Ref_Back_MPT_mean = [1.637, 1.512, 1.637, 1.680];
Ref_Back_MPT_std = [0.343, 0.404, 0.408, 0.482];

global AreaNamesData;

filePath = get_crf_sheet_path();

nArea = numel(AreaNamesData);
SubjectAreas.QST_HPT(:) = NaN;  % Otherwise by default it is 0s
SubjectAreas.QST_PPT(:) = NaN;
SubjectAreas.QST_MPT(:) = NaN;
SubjectAreas.QST_Dpain(:) = NaN;

for area = 1:nArea
    sheet = ['QST ' AreaNamesData{area}];
    Table = readtable(filePath, 'Sheet', sheet, 'DataRange', 'A2', 'VariableNamesRange', 'A1');
    Table(sum(ismissing(Table),2) == size(Table,2) - 1, :) = [];  % Remove empty rows
    Table = join(Table, SubjectAreas(SubjectAreas.Area == area, :), 'Keys', 'id');
    Areas = Table.Location;
    
    WherePrepainIs = startsWith(Table.Properties.VariableNames,'qst_prepain_');
    PrepainId = WherePrepainIs > 0;
    Default_Pain = table2array(Table(:,PrepainId));
    
    WhereHPTIs = startsWith(Table.Properties.VariableNames,'qst_hpt');
    HPTIds = WhereHPTIs > 0;
    HPTs = table2array(Table(:,HPTIds));
    HPT = sum(HPTs,2)/3;
    
    WherePPTIs = startsWith(Table.Properties.VariableNames,'qst_ppt');
    PPTIds = find(WherePPTIs > 0);   
    if ~isnan(PPTIds(2)) || ~isnan(PPTIds(3)) || ~isnan(PPTIds(4))
        PPTs = table2array(Table(:,[PPTIds(2) PPTIds(3) PPTIds(4)]));
        PPT = (nansum(PPTs,2)./sum(~isnan(PPTs),2))*98.07;
    elseif  ~isnan(PPTIds(6)) || ~isnan(PPTIds(7)) || ~isnan(PPTIds(8))
        PPTs = table2array(Table(:,[PPTIds(6) PPTIds(7) PPTIds(8)]));
        PPT = (nansum(PPTs,2)./sum(~isnan(PPTs),2))*98.07;
    else
        PPT = nan;
    end
    
    WhereMPTIs = startsWith(Table.Properties.VariableNames,'qst_mpt');
    MPTIds = WhereMPTIs > 0;
    MPTs = table2array(Table(:,MPTIds));
    MPT = prod(MPTs,2).^(1/10);
    
    for iSubject = 1:height(Table)
        id = find(strcmp(Subjects.id, Table.id{iSubject}));
        Subject_Age = Subjects.age(id);
        Subject_Gender = Subjects.Gender(id);
        
        if (strcmpi(Areas(iSubject), 'foot'))||(strcmpi(Areas(iSubject), 'leg'))|| (strcmpi(Areas(iSubject), 'thigh'))|| (strcmpi(Areas(iSubject), 'poplit'))
            if Subject_Age < 40 && Subject_Gender == 1
                QST_Score(1,1) =  (HPT(iSubject) - Ref_Foot_HPT_mean(3))/Ref_Foot_HPT_std(3);
                QST_Score(1,2) = (log10(PPT(iSubject)) - Ref_Foot_PPT_mean(3))/Ref_Foot_PPT_std(3);
                QST_Score(1,3) = (log10(MPT(iSubject)) - Ref_Foot_MPT_mean(3))/Ref_Foot_MPT_std(3);
            elseif Subject_Age < 40 && Subject_Gender == 2
                QST_Score(1,1) =  (HPT(iSubject) - Ref_Foot_HPT_mean(1))/Ref_Foot_HPT_std(1);
                QST_Score(1,2) = (log10(PPT(iSubject)) - Ref_Foot_PPT_mean(1))/Ref_Foot_PPT_std(1);
                QST_Score(1,3) = (log10(MPT(iSubject)) - Ref_Foot_MPT_mean(1))/Ref_Foot_MPT_std(1);
            elseif Subject_Age >= 40 && Subject_Gender == 1
                QST_Score(1,1) =  (HPT(iSubject) - Ref_Foot_HPT_mean(4))/Ref_Foot_HPT_std(4);
                QST_Score(1,2) = (log10(PPT(iSubject)) - Ref_Foot_PPT_mean(4))/Ref_Foot_PPT_std(4);
                QST_Score(1,3) = (log10(MPT(iSubject)) - Ref_Foot_MPT_mean(4))/Ref_Foot_MPT_std(4);
            elseif Subject_Age >= 40 && Subject_Gender == 2
                QST_Score(1,1) =  (HPT(iSubject) - Ref_Foot_HPT_mean(2))/Ref_Foot_HPT_std(2);
                QST_Score(1,2) = (log10(PPT(iSubject)) - Ref_Foot_PPT_mean(2))/Ref_Foot_PPT_std(2);
                QST_Score(1,3) = (log10(MPT(iSubject))- Ref_Foot_MPT_mean(2))/Ref_Foot_MPT_std(2);
            else
                QST_Score(1,1) = nan;
                QST_Score(1,2) = nan;
                QST_Score(1,3) = nan;
            end
        elseif (strcmpi(Areas(iSubject), 'hand'))||(strcmpi(Areas(iSubject), 'elbow'))|| (strcmpi(Areas(iSubject), 'forearm'))|| (strcmpi(Areas(iSubject), 'upperarm'))|| (strcmpi(Areas(iSubject), 'breast')) % Hand
            if Subject_Age < 40 && Subject_Gender == 1
                QST_Score(1,1) =  (HPT(iSubject) - Ref_Hand_HPT_mean(3))/Ref_Hand_HPT_std(3);
                QST_Score(1,2) = (log10(PPT(iSubject)) - Ref_Hand_PPT_mean(3))/Ref_Hand_PPT_std(3);
                QST_Score(1,3) =(log10(MPT(iSubject)) - Ref_Hand_MPT_mean(3))/Ref_Hand_MPT_std(3);
            elseif Subject_Age < 40 && Subject_Gender == 2
                QST_Score(1,1) =  (HPT(iSubject) - Ref_Hand_HPT_mean(1))/Ref_Hand_HPT_std(1);
                QST_Score(1,2) = (log10(PPT(iSubject)) - Ref_Hand_PPT_mean(1))/Ref_Hand_PPT_std(1);
                QST_Score(1,3) =(log10(MPT(iSubject)) - Ref_Hand_MPT_mean(1))/Ref_Hand_MPT_std(1);
            elseif Subject_Age >= 40 && Subject_Gender == 1
                QST_Score(1,1) =  (HPT(iSubject) - Ref_Hand_HPT_mean(4))/Ref_Hand_HPT_std(4);
                QST_Score(1,2) = (log10(PPT(iSubject)) - Ref_Hand_PPT_mean(4))/Ref_Hand_PPT_std(4);
                QST_Score(1,3) = (log10(MPT(iSubject)) - Ref_Hand_MPT_mean(4))/Ref_Hand_MPT_std(4);
            elseif Subject_Age >= 40 && Subject_Gender == 2
                QST_Score(1,1) =  (HPT(iSubject) - Ref_Hand_HPT_mean(2))/Ref_Hand_HPT_std(2);
                QST_Score(1,2) = (log10(PPT(iSubject)) - Ref_Hand_PPT_mean(2))/Ref_Hand_PPT_std(2);
                QST_Score(1,3) = (log10(MPT(iSubject)) - Ref_Hand_MPT_mean(2))/Ref_Hand_MPT_std(2);
            else
                QST_Score(1,1) = nan;
                QST_Score(1,2) = nan;
                QST_Score(1,3) = nan;
            end
            
        elseif (strcmpi(Areas(iSubject), 'back'))|| (strcmpi(Areas(iSubject), 'shoulder'))|| (strcmpi(Areas(iSubject), 'belly'))|| (strcmpi(Areas(iSubject), 'buttocks'))% back and shoulder
            
            if Subject_Age < 40 && Subject_Gender == 1
                QST_Score(1,1) =  (HPT(iSubject) - Ref_Back_HPT_mean(3))/Ref_Back_HPT_std(3);
                QST_Score(1,2) = (log10(PPT(iSubject)) - Ref_Back_PPT_mean(3))/Ref_Back_PPT_std(3);
                QST_Score(1,3) =(log10(MPT(iSubject)) - Ref_Back_MPT_mean(3))/Ref_Back_MPT_std(3);
            elseif Subject_Age < 40 && Subject_Gender == 2
                QST_Score(1,1) =  (HPT(iSubject) - Ref_Back_HPT_mean(1))/Ref_Back_HPT_std(1);
                QST_Score(1,2) = (log10(PPT(iSubject)) - Ref_Back_PPT_mean(1))/Ref_Back_PPT_std(1);
                QST_Score(1,3) =(log10(MPT(iSubject)) - Ref_Back_MPT_mean(1))/Ref_Back_MPT_std(1);
            elseif Subject_Age >= 40 && Subject_Gender == 1
                QST_Score(1,1) =  (HPT(iSubject) - Ref_Back_HPT_mean(4))/Ref_Back_HPT_std(4);
                QST_Score(1,2) = (log10(PPT(iSubject)) - Ref_Back_PPT_mean(4))/Ref_Back_PPT_std(4);
                QST_Score(1,3) = (log10(MPT(iSubject)) - Ref_Back_MPT_mean(4))/Ref_Back_MPT_std(4);
            elseif Subject_Age >= 40 && Subject_Gender == 2
                QST_Score(1,1) =  (HPT(iSubject) - Ref_Back_HPT_mean(2))/Ref_Back_HPT_std(2);
                QST_Score(1,2) = (log10(PPT(iSubject)) - Ref_Back_PPT_mean(2))/Ref_Back_PPT_std(2);
                QST_Score(1,3) = (log10(MPT(iSubject)) - Ref_Back_MPT_mean(2))/Ref_Back_MPT_std(2);
            else
                QST_Score(1,1) = nan;
                QST_Score(1,2) = nan;
                QST_Score(1,3) = nan;
            end
        else
            QST_Score(1,1) = nan;
            QST_Score(1,2) = nan;
            QST_Score(1,3) = nan;
        end
        row = strcmpi(SubjectAreas.id, Table.id{iSubject}) & SubjectAreas.Area == area;
        SubjectAreas.QST_HPT(row) = QST_Score(1);
        SubjectAreas.QST_PPT(row) = QST_Score(2);
        SubjectAreas.QST_MPT(row) = QST_Score(3);
        SubjectAreas.QST_Dpain(row) = Default_Pain(iSubject);
    end
end
end