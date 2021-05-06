clc;
clear all;close all;
warning off all;
addpath('utils');

algName = 'den_pts';
datasetPath = 'dataset\'; % dataset path
seqPath = [datasetPath 'test_data\images\']; % sequence path
detPath = ['..\results\localization\' algName '\']; % detection input path
listPath = [datasetPath 'testlist.txt']; % sequence list path
gtPath = [datasetPath 'annotations\']; % annotation path
isSeqDisplay = true; % flag to display the groundtruth and detections

% process the annotations and groundtruth
[allgt, alldet] = saveAnnoRes(gtPath, detPath, listPath);

% show the groundtruth and detection results
displaySeq(seqPath, listPath, allgt, alldet, isSeqDisplay);

% claculate average precision over all 25 distance thresholds (i.e., [1:25])
mAP = calcAccuracy(allgt, alldet);

% print the average precision and recall
disp(['Average Precision@1:25 = ' num2str(roundn(mean(mAP(1:25)),-2)) '%.']);
disp(['Average Precision@10    = ' num2str(roundn(mAP(10),-2)) '%.']);
disp(['Average Precision@15    = ' num2str(roundn(mAP(15),-2)) '%.']);
disp(['Average Precision@20    = ' num2str(roundn(mAP(20),-2)) '%.']);