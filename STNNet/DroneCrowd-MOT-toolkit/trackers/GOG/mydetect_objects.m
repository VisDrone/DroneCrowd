function [dres, bboxes] = mydetect_objects(sceneInfo, vid_path)

n_cores = 4;              %% number of cores to use. Matlab doesn't let you use more that 8 cores on a single machine. decrese it if you have less than 8 cores.
n_cores = min(8, n_cores);
if matlabpool('size') ~= n_cores
  if matlabpool('size') > 0
    matlabpool('close');
  end
  matlabpool(n_cores);
end

dirlist = dir([vid_path '*.jpg']);

frames=1:length(sceneInfo.frameNums);
[detections nPoints]=parseDetections(sceneInfo, frames); 

parfor i=1:length(dirlist)
  display(['frame ' num2str(i)]);
  im = imread([vid_path dirlist(i).name]);
  
  boxes = detect(im, model, thresh);  %% running the detector
  bbox =  getboxes(model, boxes);
  bbox
  bboxes(i).bbox = nms(bbox, 0.5);    %% running non-max-suppression to suppress overlaping weak detections.
end

dres = bboxes2dres(bboxes);           %% converting the data format.
dres.x = dres.x/2;                    %% compensate doubling image size.
dres.y = dres.y/2;
dres.w = dres.w/2;
dres.h = dres.h/2;

matlabpool('size');

