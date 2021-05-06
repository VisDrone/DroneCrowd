function [dres bboxes] = detect_objects(vid_path)
% vid_path = 'data/seq03-img-left/';
thresh = -2;              %% threshod on SVM response in human detection, we'll have more detections by decreasing it.

n_cores = 8;              %% number of cores to use. Matlab doesn't let you use more that 8 cores on a single machine. decrese it if you have less than 8 cores.
n_cores = min(8, n_cores);
if matlabpool('size') ~= n_cores
  if matlabpool('size') > 0
    matlabpool('close');
  end
  matlabpool(n_cores);
end

dirlist = dir([vid_path '*.png']);

tmp = load ('3rd_party/voc-release3.1/INRIA/inria_final.mat');  %% load the model for human. This can be changed to any of those 20 objects in PASCAL competition.
model= tmp.model;
clear tmp

parfor i=1:length(dirlist)
  display(['frame ' num2str(i)]);
  im = imread([vid_path dirlist(i).name]);
  im = imresize(im,2);                %% double the image size to detect small objects.
  
  boxes = detect(im, model, thresh);  %% running the detector
  bbox =  getboxes(model, boxes);
  
  bboxes(i).bbox = nms(bbox, 0.5);    %% running non-max-suppression to suppress overlaping weak detections.
end
dres = bboxes2dres(bboxes);           %% converting the data format.
dres.x = dres.x/2;                    %% compensate doubling image size.
dres.y = dres.y/2;
dres.w = dres.w/2;
dres.h = dres.h/2;

matlabpool('size');

