function RawTrials = Read_EEG(RawTrials)

[G, Gid, Garea, Gb] = findgroups(RawTrials.id, RawTrials.Area, RawTrials.B);
badSet = false(1, numel(Gid));
for iG = 1:numel(Gid)
    %subjectId = regexp(Gid{iG},'\d*','Match','once');
    subjectId = str2double(erase(Gid{iG},'CRPP_'));
    
    [EEG, EOG] = Read_EEG_Set(subjectId, Garea(iG), Gb(iG));
    
    % Check that we have the same number of trials on EEG and SC,
    % if not drop all the trials (could just drop SC and/or EEG instead, or
    % check how many NRS we have).
    if height(EEG) ~= sum(G == iG)
        warning('EEG and SC data incompatible for %s, area %d, B%d. Trials dropped.', Gid{iG}, Garea(iG), Gb(iG));
        badSet(iG) = true;
    else
        if ~isempty(EEG)
            RawTrials.EEG(G == iG) = split_rows(EEG);
        end
        if ~isempty(EOG)
            RawTrials.EOG(G == iG) = split_rows(EOG);
        end
    end
end
RawTrials = RawTrials(~ismember(G, find(badSet)),:);

end
