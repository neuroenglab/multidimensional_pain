function SubjectAreas = Read_CPM_Effect(Subjects, SubjectAreas)
%{

Compute and store in a pre-existing structure the conditionned pain
modulation features

%}

CPM_ExcelSheet = [ "CPM&TSP mp-cold", "CPM&TSP con-cold"];
SubjectAreas.CPM_HPT_Effect(:) = NaN;  % Otherwise by default it is 0s
SubjectAreas.CPM_HPT_PreExperiment(:) = NaN;
SubjectAreas.CPM_HPT_PostExperiment(:) = NaN;
SubjectAreas.CPM_PPT_Effect(:) = NaN;
SubjectAreas.CPM_PPT_PreExperiment(:) = NaN;
SubjectAreas.CPM_PPT_PostExperiment(:) = NaN;

for area = 1:2
    Table = Read_CRF_Data(CPM_ExcelSheet(area));
    
    WhereIsB1 = startsWith(Table.Properties.VariableNames,'tsp_ppt1_precold_');
    BeforeId1 = WhereIsB1 > 0;
    Before = table2array(Table(:,BeforeId1));
    PPT_Threshold_B =  nansum(Before,2)./sum(~isnan(Before),2);
    
    WhereIsA = startsWith(Table.Properties.VariableNames,'tsp_ppt_postcold_');
    AfterId = WhereIsA >0;
    PPT_Threshold_A = table2array(Table(:,AfterId));
    
    WhereIsB1 = startsWith(Table.Properties.VariableNames,'tsp_hpt1_precold_');
    BeforeId1 = WhereIsB1 > 0;
    Before = table2array(Table(:,BeforeId1));
    HPT_Threshold_B =  nansum(Before,2)./sum(~isnan(Before),2);
    
    WhereIsA = startsWith(Table.Properties.VariableNames,'tsp_hpt_postcold_');
    AfterId = WhereIsA > 0;
    HPT_Threshold_A = table2array(Table(:,AfterId));
    
    for iSubject = 1:height(Subjects)
        iS = strcmpi(Table.id, Subjects.id(iSubject));
        CPM_Effect(1,1) =  ((HPT_Threshold_B(iS) - HPT_Threshold_A(iS))/HPT_Threshold_B(iS))*100;
        CPM_Effect(1,2) =  ((PPT_Threshold_B(iS) - PPT_Threshold_A(iS))/PPT_Threshold_B(iS))*100;
        
        row = strcmpi(SubjectAreas.id, Table.id{iS}) & SubjectAreas.Area == area;
        SubjectAreas.CPM_HPT_Effect(row) = CPM_Effect(1);
        SubjectAreas.CPM_HPT_PreExperiment(row) = HPT_Threshold_B(iS);
        SubjectAreas.CPM_HPT_PostExperiment(row) = HPT_Threshold_A(iS);
        SubjectAreas.CPM_PPT_Effect(row) = CPM_Effect(2);
        SubjectAreas.CPM_PPT_PreExperiment(row) = PPT_Threshold_B(iS);
        SubjectAreas.CPM_PPT_PostExperiment(row) = PPT_Threshold_A(iS);
    end
end
end