function detections = nmsProcess(det, isNMS, nmsThre)
if(isNMS)
    detections = [];
    numFr = max(det(:,1));
    for i = 1:numFr
        idx = det(:,1) == i;
        curdet = det(idx,:);
        pick = nms(curdet(:,3:7), nmsThre);
        detections = cat(1, detections, curdet(pick,:));
    end    
else
    detections = det;
end