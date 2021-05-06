function displayTrackingResult(sceneInfo, res, seqName)
% Display Tracking Result
%
% Take scene information sceneInfo and
% the tracking result from stateInfo
% 
% 
% (C) Anton Andriyenko, 2012
%
% The code may be used free of charge for non-commercial and
% educational purposes, the only requirement is that this text is
% preserved within the derivative work. For any other purpose you
% must contact the authors for permission. This code may not be
% redistributed without written permission from the authors.

%% convert res to stateInfo
stateInfo = [];
stateInfo.F = max(sceneInfo.frameNums);
stateInfo.frameNums = sceneInfo.frameNums;
index = 0;
cur_id = -1;

res = sortrows(res,2);

for i = 1:size(res,1)
    if (cur_id ~= res(i,2))
        cur_id = res(i,2);
        index = index + 1;
        stateInfo.X(:,index) = zeros(stateInfo.F,1);
        stateInfo.Y(:,index) = zeros(stateInfo.F,1);
        stateInfo.Xi(:,index) = zeros(stateInfo.F,1);
        stateInfo.Yi(:,index) = zeros(stateInfo.F,1);
        stateInfo.W(:,index) = zeros(stateInfo.F,1);
        stateInfo.H(:,index) = zeros(stateInfo.F,1);
    end
    bbox = res(i,:);
    n = bbox(1);
    stateInfo.X(n,index) = bbox(3)+0.5*bbox(5);
    stateInfo.Y(n,index) = bbox(4)+bbox(6);
    stateInfo.Xi(n,index) = stateInfo.X(n,index);
    stateInfo.Yi(n,index) = stateInfo.Y(n,index);
    stateInfo.W(n,index) = bbox(5);
    stateInfo.H(n,index) = bbox(6);   
end             

reopenFig(['Tracking Results of Sequence ' seqName]);
displayBBoxes(sceneInfo, stateInfo);