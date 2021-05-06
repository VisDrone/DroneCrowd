function res = saveResTxt(bboxes_tracked)

frame_num = size(bboxes_tracked, 2);
track_bbox = zeros(1, 6);
res = [];
for frame = 1:frame_num
    bboxes = bboxes_tracked(1,frame).bbox;
    id_num = size(bboxes, 1);
    for id = 1:id_num
        track_bbox(1) = frame;
        track_bbox(2) = bboxes(id, 5);    
        track_bbox(3) = bboxes(id, 1);
        track_bbox(4) = bboxes(id, 2);
        track_bbox(5) = bboxes(id, 3) - bboxes(id, 1);
        track_bbox(6) = bboxes(id, 4) - bboxes(id, 2);
        res = cat(1, res, [track_bbox, 1, -1, -1, -1]);
    end
end

if(~isempty(res))
    res = sortrows(res,1);
end