function Trials = Read_SC(Subjects)
%{

Load in "Raw_Data" structure SCRs raw data

Input: - UserPath: Path to access the Balgrist Dataset folder
        - Dataset: Stucture containing subjective and semi subjective
       information of every subject

Output: - Raw_Data: Structure containing SCRs raw data
        - b1, b2: Size of bloc 1 and bloc 2 of the experiment

%}

nSubjects = height(Subjects);

UserPath = get_dataset_dir();

UserPath=fullfile(UserPath, '/EP_SSR/');

Trials = [];

for subject = 1:nSubjects
    id = Subjects.id(subject);
    strId = split(id, '_');
    strId = strId{2};
    
    for area = 1:3
        if area == 1
            SCH_b1 = try_read(strcat(UserPath, "CHEP/Most painful/3_SSR hand/chep_mp_ssrh_b1_crpp_", strId,".txt"));
            SCH_b2 = try_read(strcat(UserPath, "CHEP/Most painful/3_SSR hand/chep_mp_ssrh_b2_crpp_", strId,".txt"));
            SCF_b1 = try_read(strcat(UserPath, "CHEP/Most painful/4_SSR foot/chep_mp_ssrf_b1_crpp_", strId,".txt"));
            SCF_b2 = try_read(strcat(UserPath, "CHEP/Most painful/4_SSR foot/chep_mp_ssrf_b2_crpp_", strId,".txt"));
        elseif area == 2
            SCH_b1 = try_read(strcat(UserPath, "CHEP/Control/3_SSR hand/chep_con_ssrh_b1_crpp_", strId,".txt"));
            SCH_b2 = try_read(strcat(UserPath, "CHEP/Control/3_SSR hand/chep_con_ssrh_b2_crpp_", strId,".txt"));
            SCF_b1 = try_read(strcat(UserPath, "CHEP/Control/4_SSR foot/chep_con_ssrf_b1_crpp_", strId,".txt"));
            SCF_b2 = try_read(strcat(UserPath, "CHEP/Control/4_SSR foot/chep_con_ssrf_b2_crpp_", strId,".txt"));
        else
            SCH_b1 = try_read(strcat(UserPath, "CHEP/Additional/3_SSR hand/chep_ad_ssrh_b1_crpp_", strId,".txt"));
            SCH_b2 = try_read(strcat(UserPath, "CHEP/Additional/3_SSR hand/chep_ad_ssrh_b2_crpp_", strId,".txt"));
            SCF_b1 = try_read(strcat(UserPath, "CHEP/Additional/4_SSR foot/chep_ad_ssrf_b1_crpp_", strId,".txt"));
            SCF_b2 = try_read(strcat(UserPath, "CHEP/Additional/4_SSR foot/chep_ad_ssrf_b2_crpp_", strId,".txt"));
        end
        
        trimLength = false;
        if trimLength
            if ~isempty(SCH_b1) && ~isempty(SCH_b2)
                if size(SCH_b1,1) > size(SCH_b2,1)
                    SCH_b1(size(SCH_b2,1)+1:end,:) = [];
                elseif size(SCH_b1,1) < size(SCH_b2,1)
                    SCH_b2(size(SCH_b1,1)+1:end,:) = [];
                end
            end
            if ~isempty(SCF_b1) && ~isempty(SCF_b2)
                if size(SCF_b1,1) > size(SCF_b2,1)
                    SCF_b1(size(SCF_b2,1)+1:end,:) = [];
                elseif size(SCF_b1,1) < size(SCF_b2,1)
                    SCF_b2(size(SCF_b1,1)+1:end,:) = [];
                end
            end
        end
        
        if ~isempty(SCH_b1) && ~isempty(SCF_b1)
            % Expected issue with chep_mp_ssrh_b1_crpp_025.txt (bad),
            % rename it to avoid to load it
            assert(size(SCH_b1,2) == size(SCF_b1,2));
        end
        if ~isempty(SCH_b2) && ~isempty(SCF_b2)
            assert(size(SCH_b2,2) == size(SCF_b2,2));
        end
        
        Trials = append_trials(Trials, SCH_b1, SCF_b1, id, 1, area);
        Trials = append_trials(Trials, SCH_b2, SCF_b2, id, 2, area);
    end
end
end

function out = try_read(filePath)
out = [];
if isfile(filePath)
    out = table2array(readtable(filePath,'HeaderLines',3));
end
end

function [Trials, n] = append_trials(Trials, SCH, SCF, id, b, Area)
SCH = split_rows(SCH');
SCF = split_rows(SCF');
n = max(height(SCH), height(SCF));
id = repmat(id, n, 1);
B = repmat(b, n, 1);
Area = repmat(Area, n, 1);
if isempty(SCH)
    SCH = cell(n, 1);
end
if isempty(SCF)
    SCF = cell(n, 1);
end
iTrial = (1:n)';
T = table(id, Area, B, iTrial, SCH, SCF);
if isempty(Trials)
    Trials = T;
else
    Trials = [Trials; T];
end
end

