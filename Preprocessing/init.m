clear;

%% Add all subfolders to PATH
addpath(genpath(pwd));

%% Global parameters
global AreaNamesData AreaFullnames fs;
AreaNamesData = {'mp', 'con', 'ad'};  % Name of CSV files
AreaFullnames = {'Most painful', 'Control', 'Additional'};
fs = 2000;
%AreaNames = {'con','mp','ad'};
%AreaFullnames = {'Control', 'Most painful', 'Additional'};

% Avoid underscore to subscript conversion
set(groot,'defaulttextinterpreter','none');  
set(groot, 'defaultAxesTickLabelInterpreter','none');  
set(groot, 'defaultLegendInterpreter','none');
