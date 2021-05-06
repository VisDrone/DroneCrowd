close all
clear all

save_path = 'E:\CAS\多目标实验\MultiTargetTracking\MultiTargetTracking\bahnhof\PLS\bahnhof_pls_detect.mat';
Folder = 'E:\CAS\多目标实验\MultiTargetTracking\MultiTargetTracking\bahnhof\PLS\V000\';
frameNums=0:998;
fileFormat='I%05d.txt';
index = 1;

for fr=1:999
    detects=load([Folder sprintf(fileFormat,frameNums(fr))]);
    num=size(detects,1);
    for j=1:num
        dres.x(index, 1) = detects(j,1);
        dres.y(index, 1) = detects(j,2);
        dres.w(index, 1) = detects(j,3);
        dres.h(index, 1) = detects(j,4);
        dres.r(index, 1) = detects(j,5);
        dres.fr(index, 1) = fr;
        index = index + 1;
    end
end

save(save_path, 'dres');

