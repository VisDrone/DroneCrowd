clc;
clear all;close all;
warning off all;

% add toolboxes
addpath('display');
addpath('eval');
addpath(genpath('utils'));

algName = 'den_pts';
datasetPath = 'dataset\'; % dataset path
seqPath = [datasetPath 'test_data\images\']; % sequence path
detPath = ['..\results\localization\' algName '\']; % detection input path
resPath = ['..\results\tracking\' algName '\']; % result path
listPath = [datasetPath 'testlist.txt']; % sequence list path
gtPath = [datasetPath 'annotations\']; % annotation path

isSeqDisplay = false; % flag to display the detections 
trackerName = 'GOG'; % the tracker name

%% run the tracker
runTrackerAllClass(isSeqDisplay, seqPath, detPath, resPath, listPath, trackerName);

%% evaluate the tracker
[ap, recall, precision] = evaluateTrackA(listPath, resPath, gtPath, trackerName);