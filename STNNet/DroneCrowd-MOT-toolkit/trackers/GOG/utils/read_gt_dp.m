close all
clear all

save_path = 'H:\dataset\ETHZ\bahnhof\gt\bahnhof_ground_truth.mat';
load('H:\dataset\ETHZ\bahnhof\gt\bahnhof_gt.mat');
idNum = size(bahnhof.Trajectory, 1);
allNum = 999;
posNum = 0;

for i = 1:allNum
    for j = 1:idNum
        frameNum = size(bahnhof.Trajectory(j,1).Frame, 1);
        for k = 1:frameNum
            frame = bahnhof.Trajectory(j,1).Frame(k,1).ATTRIBUTE.frame_no+1;
            if (frame == i)
                posNum = posNum + 1;
                gt.x(posNum, 1) = bahnhof.Trajectory(j,1).Frame(k,1).ATTRIBUTE.x;
                gt.y(posNum, 1) = bahnhof.Trajectory(j,1).Frame(k,1).ATTRIBUTE.y;
                gt.w(posNum, 1) = bahnhof.Trajectory(j,1).Frame(k,1).ATTRIBUTE.width;
                gt.h(posNum, 1) = bahnhof.Trajectory(j,1).Frame(k,1).ATTRIBUTE.height;
                gt.fr(posNum, 1) = i;
                gt.vid(posNum, 1) = 3;
            end
        end
    end
end

save(save_path, 'gt');

