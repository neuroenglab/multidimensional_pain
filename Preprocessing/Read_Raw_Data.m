function [Subjects, SubjectAreas, RawTrials] = Read_Raw_Data()

disp('Reading subject data...');
[Subjects, SubjectAreas] = Read_Subjects();
disp('Reading trials data...');
RawTrials = Read_Raw_Trials(Subjects);

end

