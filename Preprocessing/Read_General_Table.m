function [Subjects, SubjectAreas] = Read_General_Table()
Subjects = Read_CRF_Data('General');
nSubjects = find(cellfun(@isempty, Subjects.id),1)-1;

% The other rows have only the random numbers
% For some reason it takes 5 empty columns
Subjects = Subjects(1:nSubjects,1:28);

Subjects.Subject = cellfun(@str2num, erase(Subjects.id,'CRPP_'));
Subjects = movevars(Subjects,'Subject','Before',1);

nAreas = 3;
id = repelem(Subjects.id, nAreas);
Area = repmat((1:nAreas)', numel(Subjects.id), 1);
SubjectAreas = table(id, Area);

Subjects.BMI = Subjects.weight./(Subjects.height*10^-2).^2;

[Subjects, SubjectAreas] = split_area(SubjectAreas, Subjects, 'most_pain_area', 1);
[Subjects, SubjectAreas] = split_area(SubjectAreas, Subjects, 'control_area', 2);
[Subjects, SubjectAreas] = split_area(SubjectAreas, Subjects, 'additional_area', 3);

end

function [Table, SubjectAreas] = split_area(SubjectAreas, Table, column, areaCode)
ids = find(SubjectAreas.Area == areaCode);
assert(numel(ids) == height(Table));
iToFix = strcmp(Table.(column), 'wrist-right');
Table.(column)(iToFix) = {'wrist_right'};
missingTable = cellfun(@isempty, Table.(column));
missing = ids(missingTable);
notMissingTable = find(~missingTable);
notMissing = ids(notMissingTable);
c = count(Table.(column)(notMissingTable),'_');
c1 = c == 1;
c2 = c == 2;
if any(c2)
    s = lower(split(Table.(column)(notMissingTable(c2)),'_'));
    SubjectAreas.Location(notMissing(c2)) = s(:, 1);
    SubjectAreas.VertebralLevel(notMissing(c2)) = s(:, 2);
    SubjectAreas.LeftRight(notMissing(c2)) = s(:, 3);
end
if any(c1)
    s = lower(split(Table.(column)(notMissingTable(c1)),'_'));
    if sum(c1) == 1
        % split acts silly
        SubjectAreas.Location(notMissing(c1)) = s(1);
        SubjectAreas.LeftRight(notMissing(c1)) = s(2);
    else
        SubjectAreas.Location(notMissing(c1)) = s(:, 1);
        SubjectAreas.LeftRight(notMissing(c1)) = s(:, 2);
    end
end
assert(all(c1 | c2), ['Invalid ' column ' for ' strjoin(Table.id(notMissingTable(not(c1|c2))), ', ')]);

% Translate german terms and fix errors
SubjectAreas.Location(strcmpi(SubjectAreas.Location,'gesäss')) = {'buttocks'};
SubjectAreas.Location(strcmpi(SubjectAreas.Location,'ellbow')) = {'elbow'};
SubjectAreas.Location(strcmpi(SubjectAreas.Location,'sholder')) = {'shoulder'};
SubjectAreas.Location(strcmpi(SubjectAreas.Location,'oberarm')) = {'upperarm'};
SubjectAreas.LeftRight(strcmpi(SubjectAreas.LeftRight,'rechs')) = {'right'};
SubjectAreas.LeftRight(strcmpi(SubjectAreas.LeftRight,'right (lateral)')) = {'right'};

% Otherwise gives {0x0 double}
SubjectAreas.Location(missing) = {''};
SubjectAreas.VertebralLevel(missing) = {''};
SubjectAreas.LeftRight(missing) = {''};
Table = removevars(Table, column);
end

