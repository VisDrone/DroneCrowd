function [res, runTime] = run_tracker(seqLen, detections)

%% setting parameters for tracking
c_en      = 12;     %% 10 birth cost, has no influence on the result
c_ex      = 9;     %% 10 death cost, has no influence on the result
c_ij      = 0;     %% 0 transition cost
betta     = 0.2;   %% 0.2 betta, increase will have less tracks, for every single detection
max_it    = inf;    %% inf max number of iterations (max number of tracks)
thr_cost  = 15;     %% 18 max acceptable cost for a track (increase it to have more tracks.), for every tracklet 19.8

time_start = tic;
%% Run object/human detector on all frames.
frameNums = 1:seqLen;
dres = greedy_detect_generator(detections, frameNums);

%% Running tracking algorithms
dres_dp_nms   = tracking_dp(dres, c_en, c_ex, c_ij, betta, thr_cost, max_it, 1);
dres_dp_nms.r = -dres_dp_nms.id;
runTime = toc(time_start);

%% save the tracking result
bboxes_tracked = dres2bboxes(dres_dp_nms, numel(frameNums));  %% we are visualizing the "DP with NMS in the lop" results. Can be changed to show the results of DP or push relabel algorithm.
res = saveResTxt(bboxes_tracked);