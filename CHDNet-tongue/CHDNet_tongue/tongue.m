clear all; close all; clc;
addpath('./Utils');
addpath('./Liblinear');

% load('./tongue/normal_sample.mat');% load  data
% load('./tongue/unnormal_sample.mat');% load  data

load('./tongue/normal_sample.mat');% load  data
load('./tongue/unnormal_sample.mat');% load  data

ImgSize = 32;
Averaged_TimeperTest = [];
Accuracy = [];
ErRate = [];
CHDNet_TrainTime =[];
LinearSVM_TrnTime =[];

NumNormalSamples = 40;
normalImg =length(normal_sample); 
randIdx = randperm(normalImg);

normal_train_data = [];
normal_train_lable = [];
normal_test_data = [];
normal_test_lable = [];

for i = 1:NumNormalSamples
    normal_train_data{i} = double(normal_sample{randIdx(i)});
    normal_train_lable{i} = double(normal_lable{randIdx(i)});
end

for j = 1:(normalImg - NumNormalSamples)
    normal_test_data{j} = double(normal_sample{randIdx(j+NumNormalSamples)});
    normal_test_lable{j} = double(normal_lable{randIdx(j+NumNormalSamples)});
end

NumUnnormalSamples = 44;
unnormalImg =length(unnormal_sample); 
randIdx = randperm(unnormalImg);

unnormal_train_data = [];
unnormal_train_lable = [];
unnormal_test_data = [];
unnormal_test_lable = [];

for i = 1:NumUnnormalSamples
    unnormal_train_data{i} = double(unnormal_sample{randIdx(i)});
    unnormal_train_lable{i} = double(unnormal_lable{randIdx(i)});
end

for j = 1:(unnormalImg - NumUnnormalSamples)
    unnormal_test_data{j} = double(unnormal_sample{randIdx(j+NumUnnormalSamples)});
    unnormal_test_lable{j} = double(unnormal_lable{randIdx(j+NumUnnormalSamples)});
end

TrainDataCell = [normal_train_data'; unnormal_train_data'];
TestData_ImgCell = [normal_test_data'; unnormal_test_data'];
TrainLabel = [normal_train_lable'; unnormal_train_lable'];
TrainLabel = cell2mat(TrainLabel);
TestLabels = [normal_test_lable'; unnormal_test_lable'];
TestLabels = cell2mat(TestLabels);

% ===========================================================

NumTestImg = length(TestLabels);

% We use the parameters in our submission paper
CHDNet.NumStages = 2;
CHDNet.PatchSize = [5 5];
CHDNet.NumFilters = [8 8];
CHDNet.HistBlockSize = [8 8];
CHDNet.BlkOverLapRatio = 0;
CHDNet.Pyramid = [4 2 1];

fprintf('\n ====== CHDNet Training ======= \n')
tic;
[ftrain V] = CHDNet_train(TrainDataCell,CHDNet);
CHDNet_TrainTime = toc;
clear TrainDataCell;
fprintf('\n ====== CHDNet Feature Extraction ======= \n')
ftest = CHDNet_Test_Fea(TestData_ImgCell,V,CHDNet);
fea = [ftrain';ftest'];
labels = [TrainLabel;TestLabels];
fprintf('\n ====== Saving Feature and Label File ======= \n')
csvwrite('feature_CHDNet.csv',full(fea));
csvwrite('label_CHDNet.csv',labels);

