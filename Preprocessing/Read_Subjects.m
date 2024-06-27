function [Subjects, SubjectAreas] = Read_Subjects()
[Subjects, SubjectAreas] = Read_General_Table();
Subjects = Read_Questionnaires(Subjects);
SubjectAreas = Read_QST(Subjects, SubjectAreas);
SubjectAreas = Read_CPM_Effect(Subjects, SubjectAreas);
SubjectAreas = Read_TS_Slope(Subjects, SubjectAreas);
end

