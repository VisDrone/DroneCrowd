function [detections nDets]=cutDetections(detections,nDets)
% remove all detections that are
% outside the tracking area
% 
% (C) Anton Andriyenko, 2012
%
% The code may be used free of charge for non-commercial and
% educational purposes, the only requirement is that this text is
% preserved within the derivative work. For any other purpose you
% must contact the authors for permission. This code may not be
% redistributed without written permission from the authors.
%

global sceneInfo opt
if opt.track3d && opt.cutToTA
    F=length(detections);
    Field = fieldnames(detections);
    nDets=0;
    for t=1:F
        tokeep=find(detections(t).xw>=sceneInfo.trackingArea(1) & ...
                    detections(t).xw<=sceneInfo.trackingArea(2) & ...
                    detections(t).yw>=sceneInfo.trackingArea(3) & ...
                    detections(t).yw<=sceneInfo.trackingArea(4));
                
                nDets=nDets+length(tokeep);

        for iField = 1:length(Field)
            fcontent=detections(t).(char(Field(iField)));
            fcontent=fcontent(tokeep);
            detections(t).(char(Field(iField)))=fcontent;
        end
    end
end
end
