clear;clc;
addpath('../../input_trans');

outPath1 = '.\pls_detect\PETS\';
video_name = 'PETS2009-S2L1';
outName1 = [video_name '_pls_detect.mat'];
out1 = [outPath1 outName1];

global sceneInfo opt detections nPoints;
frames=1:795;
sceneInfo=getSceneInfoDCDemo;
opt.cutToTA=0;
[detections, nPoints]=parseDetections(sceneInfo,frames); 
[detections, nPoints]=cutDetections(detections,nPoints);

frameNum = size(detections,2);
count = 1;

for i = 1:frameNum
    i
    obj_num = size(detections(i).xi, 2);
    for j=1:obj_num
        dres.w(count,1) = detections(i).wd(1,j);
        dres.h(count,1) = detections(i).ht(1,j);
        w = dres.w(count,1);
        h = dres.h(count,1);
        dres.x(count,1) = detections(i).xi(1,j) - 0.5*w;
        dres.y(count,1) = detections(i).yi(1,j) - h;
        dres.fr(count,1) = i;
        dres.r(count,1) = detections(i).sc(1,j)*3-1.5;        
        count = count+1;
    end
end

save(out1,'dres');