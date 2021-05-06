function stateInfo = saveStateInfo(bboxes_tracked)

frame_num = size(bboxes_tracked, 2);
index = 1;

for frame = 1:frame_num
    bboxes = bboxes_tracked(1,frame).bbox;
    id_num = size(bboxes, 1);
    for id = 1:id_num
        track_result(index,1) = bboxes(id, 1);
        track_result(index,2) = bboxes(id, 2);
        track_result(index,3) = bboxes(id, 3) - bboxes(id, 1);
        track_result(index,4) = bboxes(id, 4) - bboxes(id, 2);
        track_result(index,5) = frame;
        track_result(index,6) = bboxes(id, 5);    
        index = index + 1;
    end
end

sorted_result = sortrows(track_result,6);

stateInfo.F = frame_num;
stateInfo.frameNums = 1:frame_num;
detect_num = size(sorted_result,1);
index = 0;
cur_id = -1;

for i = 1:detect_num
    if (cur_id ~= sorted_result(i,6))
        cur_id = sorted_result(i,6);
        index = index + 1;
        stateInfo.X(:,index) = zeros(frame_num,1);
        stateInfo.Y(:,index) = zeros(frame_num,1);
        stateInfo.Xi(:,index) = zeros(frame_num,1);
        stateInfo.Yi(:,index) = zeros(frame_num,1);
        stateInfo.W(:,index) = zeros(frame_num,1);
        stateInfo.H(:,index) = zeros(frame_num,1);
    end
    bbox = sorted_result(i,:);
    n = bbox(1,5);
    stateInfo.X(n,index) = bbox(1,1)+0.5*bbox(1,3);
    stateInfo.Y(n,index) = bbox(1,2)+bbox(1,4);
    stateInfo.Xi(n,index) = stateInfo.X(n,index);
    stateInfo.Yi(n,index) = stateInfo.Y(n,index);
    stateInfo.W(n,index) = bbox(1,3);
    stateInfo.H(n,index) = bbox(1,4);   
end             