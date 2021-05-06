clear;clc;
addpath('../../input_trans');

global sceneInfo opt;

dbfolder='E:\ComputerVision\Multi-target Tracking\TPAMI_Dataset\Pedestrian_Dataset\PETS2009\S2_L1';
video_name = 'PETS2009-S2L1';
opt.track3d = 1;

%% camera
cameraconffile=[];
if opt.track3d
     cameraconffile=fullfile(dbfolder,'camera','CameraView001.xml');
end
sceneInfo.camFile=cameraconffile;

if ~isempty(sceneInfo.camFile)
    sceneInfo.camPar=parseCameraParameters(sceneInfo.camFile);
end

%% ground truth
sceneInfo.gtFile='';
sceneInfo.gtFile=fullfile(dbfolder,'gt','PETS2009-S2L1.xml');
outPath = '.\gtInfo\PETS\';
outName = [video_name '_gtInfo.mat'];

outPath2 = '.\detect_gt\PETS\';
outName2 = [video_name '_detect_gt.mat'];
global gtInfo
sceneInfo.gtAvailable=0;
if ~isempty(sceneInfo.gtFile)
    sceneInfo.gtAvailable=1;
    % first determine the type
    [~, ~, fileext]=fileparts(sceneInfo.gtFile);
    
    if strcmpi(fileext,'.xml') % CVML
        gtInfo=parseGT(sceneInfo.gtFile);
    elseif strcmpi(fileext,'.mat')
        % check for the var gtInfo
        fileInfo=who('-file',sceneInfo.gtFile);
        varExists=0; cnt=0;
        while ~varExists && cnt<length(fileInfo)
            cnt=cnt+1;
            varExists=strcmp(fileInfo(cnt),'gtInfo');
        end
        
        if varExists
            load(sceneInfo.gtFile,'gtInfo');
        else
            warning('specified file does not contained correct ground truth');
            sceneInfo.gtAvailable=0;
        end
    end
    
    if opt.track3d
        if ~isfield(gtInfo,'Xgp') || ~isfield(gtInfo,'Ygp')
            [gtInfo.Xgp, gtInfo.Ygp]=projectToGroundPlane(gtInfo.X, gtInfo.Y, sceneInfo);
        end
    end
    
    %     if strcmpi(fileext,'.xml'),     save(fullfile(pathtogt,[gtfile '.mat']),'gtInfo'); end
end

save([outPath outName],'gtInfo');


%% _detect_gt
global gt
count = 1;
[r, c] = size(gtInfo.X);
for i=1:r
    for j=1:c
        if gtInfo.W(i,j) ~= 0 && gtInfo.H(i,j) ~= 0
            gt.x(count,1) = gtInfo.X(i,j)-0.5*gtInfo.W(i,j);
            gt.y(count,1) = gtInfo.Y(i,j)-gtInfo.H(i,j);
            gt.w(count,1) = gtInfo.W(i,j);
            gt.h(count,1) = gtInfo.H(i,j);
            gt.fr(count,1) = i;
            gt.vid(count,1) = 3;
        count = count+1;
        end
    end
end

save([outPath2 outName2],'gt');