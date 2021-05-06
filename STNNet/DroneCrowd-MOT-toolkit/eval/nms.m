function pick = nms(boxes, threshold)
% boxes:      m x 5, indicates m detections with the format [left top w h score]
% threshold:  the threshold of IOU score

if isempty(boxes)
  pick = [];
  return;
end

% get the detections
x1 = boxes(:,1);
y1 = boxes(:,2);
x2 = boxes(:,1) + boxes(:,3) - 1;
y2 = boxes(:,2) + boxes(:,4) - 1;
s = boxes(:,5);

% calculate the area of detections
area = (x2-x1+1) .* (y2-y1+1);

% sort the detections
[~, I] = sort(s);

% init
pick = s*0;
counter = 1;

% remove the redundant detections
while ~isempty(I)
    last = length(I); % the number of left detections
    i = I(last);% pick the detection with the maximal confidense score
    pick(counter) = i;
    counter = counter + 1;  

    % calculate IOU score
    xx1 = max(x1(i), x1(I(1:last-1)));
    yy1 = max(y1(i), y1(I(1:last-1)));
    xx2 = min(x2(i), x2(I(1:last-1)));
    yy2 = min(y2(i), y2(I(1:last-1)));  
    w = max(0.0, xx2-xx1+1);
    h = max(0.0, yy2-yy1+1); 
    inter = w.*h;
    o = inter ./ (area(i) + area(I(1:last-1)) - inter);

    % keep the detections if the IOU score is less than the threshold
    I = I(o<=threshold);
end
pick = pick(1:(counter-1));
