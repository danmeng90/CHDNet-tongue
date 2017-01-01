clear all; close all; clc;
addpath('./Utils');
addpath('./Liblinear');

load('./tongue/S_A_O.mat');% load  data

ImgSize = 32;
Averaged_TimeperTest = [];
Accuracy = [];
ErRate = [];
CHDNet_TrainTime =[];
LinearSVM_TrnTime =[];

%superficial 
NumSuperficialSamples = 84;%ǳ����θ��ѵ�������ĸ���
superficialImg =length(superficial_sample); %ǳ����θ��������С
randIdx = randperm(superficialImg);%�������һ���ǳ����θ��������Сһ����������

superficial_train_data = [];
superficial_train_label = [];

%ǳ����θ�׵�ѵ����������ǩ
for i = 1:NumSuperficialSamples
    superficial_train_data{i} = double(superficial_sample{randIdx(i)});
    superficial_train_label{i} = double(superficial_label{randIdx(i)});
end

%ή����θ�׵Ĳ�����������ǩ
NumAtrophicSamples = 144;
atrophicImg =length(atrophic_sample); %ή����θ��������С
randIdx = randperm(atrophicImg);%�������һ���ή����θ��������Сһ����������

atrophic_train_data = [];
atrophic_train_label = [];

%ή����θ�׵�ѵ����������ǩ
for i = 1:NumAtrophicSamples
    atrophic_train_data{i} = double(atrophic_sample{randIdx(i)});
    atrophic_train_label{i} = double(atrophic_label{randIdx(i)});
end

%��������Ĳ�����������ǩ
NumOtherSamples = 39;
otherImg =length(other_sample); %ή����θ��������С
randIdx = randperm(otherImg);%�������һ���ή����θ��������Сһ����������

other_train_data = [];
other_train_label = [];

%���������ѵ����������ǩ
for i = 1:NumOtherSamples
    other_train_data{i} = double(other_sample{randIdx(i)});
    other_train_label{i} = double(other_label{randIdx(i)});
end


%Trainset
TrainDataCell = [superficial_train_data'; atrophic_train_data'; other_train_data'];
%TrainDataCell = [superficial_train_data'; atrophic_train_data'];
TrainLabelCell = [superficial_train_label'; atrophic_train_label'; other_train_label'];
%TrainLabelCell = [superficial_train_label'; atrophic_train_label'];

TrainData = [];
TrainLabel = [];

NumTrainSample = length(TrainDataCell);
randIdx = randperm(NumTrainSample);
for i = 1 : NumTrainSample
    TrainData{i} = double(TrainDataCell{randIdx(i)});
    TrainLabel{i} = double(TrainLabelCell{randIdx(i)});
end
TrainData = TrainData';
TrainLabel = TrainLabel';
TrainLabel = cell2mat(TrainLabel);

% ===========================================================
% We use the parameters in our submission paper
CHDNet.NumStages = 2;
CHDNet.PatchSize = [5 5];
CHDNet.NumFilters = [8 8];
CHDNet.HistBlockSize = [8 8];
CHDNet.BlkOverLapRatio = 0;
CHDNet.Pyramid = [4 2 1];

fprintf('\n ====== CHDNet Parameters ======= \n')
CHDNet

fprintf('\n ====== CHDNet Training ======= \n')
tic;
[ftrain V] = CHDNet_train(TrainData,CHDNet);
CHDNetTrainTime = toc;

fprintf('\n ====== Training Linear SVM Classifier ======= \n');  
tic;  
model = train(TrainLabel, ftrain', '-s 0 -c 1 -w1 2.1538 -w0 3.6923 -w-1 1 -v 10 -q');  
LinearSVM_TrnTime = toc;  





