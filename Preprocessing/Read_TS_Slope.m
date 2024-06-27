function SubjectAreas = Read_TS_Slope(Subjects, SubjectAreas)
%{

Compute and store in a pre-existing structure different paramaters caracterising
the conditionned pain modulation and temporal summation experiment.

%}

CPM_ExcelSheet = [ "CPM&TSP mp-cold", "CPM&TSP con-cold"];
SubjectAreas.TS_HPT_Slope_Variation(:) = NaN;  % Otherwise by default it is 0s
SubjectAreas.TS_HPT_Slope_Before(:) = NaN;
SubjectAreas.TS_HPT_Reg_Variation(:) = NaN;
SubjectAreas.TS_HPT_Reg_Before(:) = NaN;
SubjectAreas.TS_HPT_Magnitude_Before(:) = NaN;
SubjectAreas.TS_HPT_Magnitude_Variation(:) = NaN;
SubjectAreas.TS_HPT_AUC_Before(:) = NaN;
SubjectAreas.TS_HPT_AUC_Variation(:) = NaN;
SubjectAreas.TS_HPT_NRS_PreExperiment = cell(height(SubjectAreas), 1);
SubjectAreas.TS_HPT_NRS_PostExperiment = cell(height(SubjectAreas), 1);

for area = 1:2
    Table = Read_CRF_Data(CPM_ExcelSheet(area));
    
    WhereIsB1 = startsWith(Table.Properties.VariableNames,'tsp_heat_stim_');
    
    BeforeId1 = find(WhereIsB1 >0);
    PreId = BeforeId1(1:12);
    Before = table2array(Table(:,PreId));
    PostId = BeforeId1(13:24);
    After = table2array(Table(:, PostId));
    
    for iSubject = 1:height(Subjects)
        iS = find(strcmpi(Subjects.id(iSubject), Table.id));
        
        %slope
        yB = Before(iS,:);
        yB(yB==999) = NaN;
        yA =  After(iS,:);
        yA(yA==999) = NaN;
        if all(isnan(yB)) || all(isnan(yA))
            continue;
        end
        xB = [1:1:12];
        maxyB = nanmax(yB);
        SlopeB = (maxyB-yB(1))/34;
        
        xA = [1:1:12];
        maxyA = nanmax(yA);
        SlopeA = (maxyA-yA(1))/34;
        Slope_Variation = SlopeA - SlopeB;
        
        %regression
        RB = fitlm(xB,yB);
        coefsB = table2array(RB.Coefficients);
        yCalcB = coefsB(2,1);
        RA = fitlm(xA,yA);
        coefsA = table2array(RA.Coefficients);
        yCalcA = coefsA(2,1);
        Reg_Variation = yCalcA - yCalcB;
        
        % Magnitude
        minyB = nanmin(yB);
        minyA = nanmin(yA);
        MagnitudeB = maxyB - minyB;
        MagnitudeA = maxyA - minyA;
        Magnitude_Variation = MagnitudeA - MagnitudeB;
        
        % Area under curve
        yB(isnan(yB)) = [];
        yA(isnan(yA)) = [];
        AUCB = trapz(yB);
        AUCA = trapz(yA);
        AUC_Variation = AUCA - AUCB;
        
        row = strcmpi(SubjectAreas.id, Table.id{iS}) & SubjectAreas.Area == area;
        SubjectAreas.TS_HPT_Slope_Variation(row) = Slope_Variation;
        SubjectAreas.TS_HPT_Slope_Before(row) = SlopeB;
        SubjectAreas.TS_HPT_Reg_Variation(row) = Reg_Variation;
        SubjectAreas.TS_HPT_Reg_Before(row) = yCalcB;
        SubjectAreas.TS_HPT_Magnitude_Before(row) = MagnitudeB;
        SubjectAreas.TS_HPT_Magnitude_Variation(row) = Magnitude_Variation;
        SubjectAreas.TS_HPT_AUC_Before(row) = AUCB;
        SubjectAreas.TS_HPT_AUC_Variation(row) = AUC_Variation;
        SubjectAreas.TS_HPT_NRS_PreExperiment(row) = {yB};
        SubjectAreas.TS_HPT_NRS_PostExperiment(row) = {yA};
    end
end
end