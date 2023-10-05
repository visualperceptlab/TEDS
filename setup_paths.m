function setup_paths()

[pathstr, ~, ~] = fileparts(mfilename('fullpath'));

% Tracker implementation
addpath(genpath([pathstr '/implementation/']));

% Utilities
addpath([pathstr '/utils/']);

% The feature extraction
addpath(genpath([pathstr '/feature_extraction/']));

% Matconvnet
addpath([pathstr '/external_libs/matconvnet/matlab/mex/']);
addpath([pathstr '/external_libs/matconvnet/matlab']);
addpath([pathstr '/external_libs/matconvnet/matlab/simplenn']);
vl_setupnn;

% PDollar toolbox
addpath(genpath([pathstr '/external_libs/pdollar_toolbox/channels']));

% Mtimesx
addpath([pathstr '/external_libs/mtimesx/']);

% mexResize
addpath([pathstr '/external_libs/mexResize/']);

% add networks
%addpath('Path to Resnet-50, from http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat');
addpath(genpath([pathstr '/feature_extraction/lookup_tables/']));