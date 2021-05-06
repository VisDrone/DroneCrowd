function [aps, recall, precision] = evaluateTrackA(listPath, resPath, gtPath, trackerName)

defaultIOUthr = 25;
defaultTrackThr = [0.10, 0.15, 0.20];

allSequences = load(listPath); % find the sequence list
num_vids = length(allSequences);

gts = cell(1,num_vids);
gt_track_labels = cell(1,num_vids);
gt_track_bboxes = cell(1,num_vids);
gt_track_thr = cell(1,num_vids);
gt_track_img_ids = cell(1,num_vids);
gt_track_generated = cell(1,num_vids);
num_track_per_class = 0;
num_imgs = 300;

fprintf('loading groundtruth\n');
for v = 1:num_vids
    seqID = allSequences(v);
    sequenceName = sprintf('%05d',seqID);
    % parse groudtruth
    gtFilename = fullfile(gtPath, [sequenceName '.mat']);
    if(~exist(gtFilename, 'file'))
        error('No annotation files is provided for evaluation.');
    end    
    clean_gtFilename = fullfile(gtPath, [sequenceName '_clean.txt']);
    if(~exist(clean_gtFilename, 'file'))   
        tmp = load(gtFilename);
        anno = tmp.anno;
        rec = [anno(:,1)+1, anno(:,2:4), repmat([20, 20, 1, 1, 0, 0], [size(anno,1), 1])];           
        dlmwrite(clean_gtFilename, rec);
    else
        rec = load(clean_gtFilename);
    end
    gts{v} = rec;
    
    tracks = [];
    num_tracks = 0;
    recs = cell(1,num_imgs);

    for i = 1:num_imgs
        idx = rec(:,1) == i;
        currec = rec(idx, :);
        recs{i} = currec;
        
        for j = 1:size(currec, 1)
            trackid = currec(j,2);
            if(isempty(find(tracks == trackid, 1)))
                num_tracks = num_tracks + 1;
                tracks = cat(1, tracks, trackid);
                num_track_per_class = num_track_per_class + 1;
            end
        end
        if(num_tracks == 0)
            continue;
        end
    end
        
    gt_track_labels{v} = ones(1,num_tracks) * -1;
    gt_track_bboxes{v} = cell(1,num_tracks);
    gt_track_thr{v} = cell(1,num_tracks);
    gt_track_img_ids{v} = cell(1,num_tracks);
    gt_track_generated{v} = cell(1,num_tracks);
    count = 0;
    for i = 1:num_imgs
        count = count + 1;
        currec = recs{count};
        for j = 1:size(currec, 1)
            trackid = currec(j,2);
            k = find(tracks == trackid);
            gt_track_img_ids{v}{k}(end+1) = i;
            if(gt_track_labels{v}(k) == -1)
                gt_track_labels{v}(k) = 1;
            end
            bb = [currec(j,3), currec(j,4), currec(j,5)+currec(j,3)-1, currec(j,6)+currec(j,4)-1];
            gt_track_bboxes{v}{k}(:,end+1) = bb;
            gt_track_thr{v}{k}(end+1) = defaultIOUthr;
        end
    end
end

fprintf('loading tracking results\n');
track_img_ids = cell(1,num_vids);
track_labels = cell(1,num_vids);
track_confs = cell(1,num_vids);
track_bboxes = cell(1,num_vids);
for v = 1:num_vids
    seqID = allSequences(v);
    sequenceName = sprintf('%05d',seqID);
    % retrieve results for current video.
    resFilename = fullfile(resPath, [sequenceName '_' trackerName '.txt']);
    resdata = load(resFilename);
    if(size(resdata,1)==0)
        continue;
    end
 
    vid_img_ids = resdata(:,1);
    vid_obj_labels = resdata(:,8);
    vid_track_ids = resdata(:,2);
    vid_obj_confs = resdata(:,7);
    vid_obj_bboxes = [resdata(:,3), resdata(:,4), resdata(:,5)+resdata(:,3)-1, resdata(:,6)+resdata(:,4)-1]';
 
    % get result for each tracklet in a video.
    track_ids = unique(vid_track_ids);
    num_tracks = length(track_ids);
    track_img_ids{v} = cell(1,num_tracks);
    track_labels{v} = ones(1,num_tracks) * -1;
    track_confs{v} = zeros(1,num_tracks);
    track_bboxes{v} = cell(1,num_tracks);
    count = 0;
    for k = track_ids'
        ind = vid_track_ids == k;
        count = count + 1;
        track_img_ids{v}{count} = vid_img_ids(ind);
        track_label = unique(vid_obj_labels(ind));
        if(numel(track_label) > 1)
            error('Find multiple labels in a tracklet.');
        end        
        track_labels{v}(count) = track_label;
        % use the mean score as a score for a tracklet.
        track_confs{v}(count) = mean(vid_obj_confs(ind));
        track_bboxes{v}{count} = vid_obj_bboxes(:,ind);
    end
end

for v = 1:num_vids
    [track_confs{v}, ind] = sort(track_confs{v},'descend');
    track_img_ids{v} = track_img_ids{v}(ind);
    track_labels{v} = track_labels{v}(ind);
    track_bboxes{v} = track_bboxes{v}(:,ind);
end
tp_cell = cell(1,num_vids);
fp_cell = cell(1,num_vids);

fprintf('accumulating\n');
num_track_thr = length(defaultTrackThr);
% iterate over videos
for v = 1:num_vids    
    num_tracks = length(track_labels{v});
    num_gt_tracks = length(gt_track_labels{v});

    tp = cell(1,num_track_thr);
    fp = cell(1,num_track_thr);
    gt_detected = cell(1,num_track_thr);
    for o = 1:num_track_thr
        tp{o} = zeros(1,num_tracks);
        fp{o} = zeros(1,num_tracks);
        gt_detected{o} = zeros(1,num_gt_tracks);
    end

    for m = 1:num_tracks
        img_ids = track_img_ids{v}{m};
        label = track_labels{v}(m);
        bboxes = track_bboxes{v}{m};
        num_obj = length(img_ids);

        ovmax = ones(1,num_track_thr) * -inf;
        kmax = ones(1,num_track_thr) * -1;
        for n = 1:num_gt_tracks
            gt_label = gt_track_labels{v}(n);
            if(label ~= gt_label)
                continue;
            end
            gt_img_ids = gt_track_img_ids{v}{n};
            gt_bboxes = gt_track_bboxes{v}{n};
            gt_thr = gt_track_thr{v}{n};

            num_matched = 0;
            num_total = length(union(img_ids, gt_img_ids));
            for j = 1:num_obj
                id = img_ids(j);
                k = find(gt_img_ids == id);
                if(isempty(k))
                    continue; % just ignore this detection if it does not belong to the evaluated object category
                end
                bb = bboxes(:,j);
                bbgt = gt_bboxes(:,k);

                dist = sqrt(((bb(1)+bb(3))/2-(bbgt(1)+bbgt(3))/2)^2 + ((bb(2)+bb(4))/2-(bbgt(2)+bbgt(4))/2)^2);
                if(dist<= gt_thr(k))
                    num_matched = num_matched + 1;
                end
            end
            ov = num_matched / num_total;
            for o = 1:num_track_thr
                if(gt_detected{o}(n))
                    continue;
                end
                if(ov >= defaultTrackThr(o) && ov > ovmax(o))
                    ovmax(o) = ov;
                    kmax(o) = n;
                end
            end
        end

        for o = 1:num_track_thr
            if(kmax(o) > 0)
                tp{o}(m) = 1;
                gt_detected{o}(kmax(o)) = 1;
            else
                fp{o}(m) = 1;
            end
        end
    end
    % put back into global vector
    tp_cell{v} = tp;
    fp_cell{v} = fp;
end

% calculate APs
[aps, recall, precision] = calcAP(track_confs, tp_cell, fp_cell, num_vids, num_track_per_class, num_track_thr, defaultTrackThr);