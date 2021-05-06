function [newgt, newdet] = dropObjectsInIgr(gt, det, imgHeight, imgWidth)
%% drop objects in ignored region
newgt = [];
newdet = [];
numFr = max(gt(:,1));
for fr = 1:numFr
    % parse objects
    idxGtFr = gt(:,1) == fr & gt(:, 8) ~= 0;
    curgt = gt(idxGtFr,:);
    idxDetFr = det(:,1) == fr;
    curdet = det(idxDetFr,:);    
    % parse ignored regions
    idxIgr = gt(:,1) == fr & gt(:, 8) == 0;
    igrRegion = gt(idxIgr, 3:6);
    if(~isempty(igrRegion))
        igrMap = zeros(imgHeight, imgWidth);
        numIgr = size(igrRegion,1);
        for j = 1:numIgr
            igrMap(igrRegion(j,2):min(imgHeight,igrRegion(j,2)+igrRegion(j,4)),igrRegion(j,1):min(imgWidth,igrRegion(j,1)+igrRegion(j,3))) = 1;
        end
        intIgrMap = createIntImg(double(igrMap));
        idxLeft = [];
        for i = 1:size(curgt, 1)
            pos = max(1,round(curgt(i,3:6)));
            x = max(1, min(imgWidth, pos(1)));
            y = max(1, min(imgHeight, pos(2)));
            w = pos(3);
            h = pos(4);
            tl = intIgrMap(y, x);
            tr = intIgrMap(y, min(imgWidth,x+w));
            bl = intIgrMap(max(1,min(imgHeight,y+h)), x);
            br = intIgrMap(max(1,min(imgHeight,y+h)), min(imgWidth,x+w));
            igrVal = tl + br - tr - bl; 
            if(igrVal/(h*w)<0.5)
                idxLeft = cat(1, idxLeft, i);
            end
        end
        
        curdet = curdet(idxLeft, :);
        idxLeft = [];
        for i = 1:size(curdet, 1)
            pos = max(1,round(curdet(i,3:6)));
            x = max(1, min(imgWidth, pos(1)));
            y = max(1, min(imgHeight, pos(2)));
            w = pos(3);
            h = pos(4);
            tl = intIgrMap(y, x);
            tr = intIgrMap(y, min(imgWidth,x+w));
            bl = intIgrMap(max(1,min(imgHeight,y+h)), x);
            br = intIgrMap(max(1,min(imgHeight,y+h)), min(imgWidth,x+w));
            igrVal = tl + br - tr - bl; 
            if(igrVal/(h*w)<0.5)
                idxLeft = cat(1, idxLeft, i);
            end
        end
        curdet = curdet(idxLeft, :);        
    end
    newgt = cat(1, newgt, curgt);
    newdet = cat(1, newdet, curdet);
    
end