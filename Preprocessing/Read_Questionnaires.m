function Subjects = Read_Questionnaires(Subjects)
%{

Load "Redcap_Fragebîgen" tables and create a structure "Dataset" containing
different subjective information for every subject.

Input: - UserPath: Path to access the Balgrist Dataset folder

Output: - Dataset: Structure containing for every subject: Subject Id, Age,
        Gender, Body Mass Index, HADS scores, PCS score, PSEQ score, Fatigue and MAIA scores

%}

UserPath = get_dataset_dir();
paths = {'KFSPCRPS_20220127.xlsx', ...
    'KFSPLBPControl_DATA_2022-02-01_1846.csv', ...
    'KFSPHealthySubjects_DATA_2022-02-01_1837', ...
    'KFSPLBP_DATA_2022-01-27_1319.csv', ...
    'KFSPSCINP_DATA_2022-01-27_1411.csv'};

cohortNames = {'crps', 'lbp', 'hc', 'lbp', 'sci_np'};

for iCohort = 1:numel(paths)
    Table = readtable(fullfile(UserPath, 'Questionnaires', paths{iCohort}));
    
    Table(:,all(ismissing(Table))) = [];
    Table.Properties.VariableNames = erase(Table.Properties.VariableNames, ['_' cohortNames{iCohort}]);
    Table.Properties.VariableNames = erase(Table.Properties.VariableNames, '_con');

    if ismember('redcap_event_name', Table.Properties.VariableNames)
        % In case of LBP
        Table = Table(strcmp(Table.redcap_event_name, 'baseline_arm_1') ,:);
        Table.Properties.VariableNames{'sex'} = 'gender';
    end

    ids = cellfun(@(id) find(strcmpi(id, Subjects.id)), Table.record_id);
    
    Subjects.age(ids) = Table.age;
    
    Subjects.Gender(ids) = Table.gender;
    
    if ~ismember(iCohort, [2 3])
        Subjects.NRS_now(ids) = Table.pain_now;
        Subjects.NRS_avg4wk(ids) = Table.pain_avg4wk;
        Subjects.NRS_max4wk(ids) = Table.pain_max4wk;
    else
        Subjects.NRS_now(ids) = 0;
        Subjects.NRS_avg4wk(ids) = 0;
        Subjects.NRS_max4wk(ids) = 0;
    end
    
    HADS_AllQuestions_D = Table{:,{'hads_2','hads_4','hads_6','hads_8','hads_10', 'hads_12','hads_14'}};
    HADS_AllQuestions_A = Table{:,{'hads_1','hads_3','hads_5','hads_7','hads_9', 'hads_11','hads_13'}};
    
    % AllQuestions can also be imported as cells with split_rows
    Subjects.HADS_AllQuestions_D(ids, :) = HADS_AllQuestions_D;
    Subjects.HADS_AllQuestions_A(ids, :) = HADS_AllQuestions_A;
    Subjects.HADS_D(ids) = sum(HADS_AllQuestions_D,2);
    Subjects.HADS_A(ids) = sum(HADS_AllQuestions_A,2);
    
    PCS_AllQuestions = Table{:,{'pcs_1','pcs_2','pcs_3','pcs_4',...
    'pcs_5','pcs_6','pcs_7','pcs_8', 'pcs_9','pcs_10',...
    'pcs_11', 'pcs_12','pcs_13'}};
    Subjects.PCS_AllQuestions(ids, :) = PCS_AllQuestions;
    Subjects.PCS(ids) = sum(PCS_AllQuestions,2);
    
    PSEQ_AllQuestions = Table{:,{'pseq_1','pseq_2', 'pseq_3', 'pseq_4'}};
    Subjects.PSEQ_AllQuestions(ids, :) = PSEQ_AllQuestions;
    Subjects.PSEQ(ids) = 24 - sum(PSEQ_AllQuestions,2);
    
    MAIA_AllQuestions = Table{:,{'maia_1','maia_2','maia_3','maia_4',...
        'maia_5','maia_6','maia_7','maia_8', 'maia_9','maia_10',...
        'maia_11', 'maia_12','maia_13','maia_14', 'maia_15','maia_16',...
        'maia_17', 'maia_18','maia_19','maia_20', 'maia_21','maia_22',...
        'maia_23', 'maia_24','maia_25','maia_26', 'maia_27','maia_28',...
        'maia_29', 'maia_30','maia_31','maia_32'}};
    for question = 5:9
        MAIA_AllQuestions(:,question) = 5 - MAIA_AllQuestions(:,question);
    end
    Subjects.MAIA_AllQuestions(ids, :) = MAIA_AllQuestions;
    Subjects.MAIA_Noticing(ids) = sum(MAIA_AllQuestions(:,1:4),2)./4;
    Subjects.MAIA_NotDistracting(ids) = sum(MAIA_AllQuestions(:,5:7),2)./3;
    Subjects.MAIA_NotWorrying(ids) = sum(MAIA_AllQuestions(:,8:10),2)./3;
    Subjects.MAIA_AttentionRegulation(ids) = sum(MAIA_AllQuestions(:,11:17),2)./7;
    Subjects.MAIA_EmotionalAwareness(ids) = sum(MAIA_AllQuestions(:,18:22),2)./5;
    Subjects.MAIA_SelfRegulation(ids) = sum(MAIA_AllQuestions(:,23:26),2)./4;
    Subjects.MAIA_BodyListening(ids) = sum(MAIA_AllQuestions(:,27:29),2)./3;
    Subjects.MAIA_Trusting(ids) = sum(MAIA_AllQuestions(:,30:32),2)./3;
    
    Subjects.Fatigue(ids) = Table.fatigue;
end

Subjects = removevars(Subjects, 'gender');  % 'Gender' stays

end

