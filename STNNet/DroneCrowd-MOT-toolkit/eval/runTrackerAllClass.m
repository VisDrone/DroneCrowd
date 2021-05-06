function runTrackerAllClass(isSeqDisplay, seqPath, detPath, resPath, listPath, trackerName)

createPath(resPath); % create the saving path of the tracking results
seqList = load(listPath); % find the sequence list
speed = 0;
numfrs = 0;

% evaluate each sequence
for idSeq = 1:length(seqList)
    if(mod(idSeq,10)==0)
        disp(['tracking the sequence ' num2str(idSeq) '/' num2str(length(seqList)) '...']);
    end
    % load detections and sequence
    seqID = seqList(idSeq);
    seqName = sprintf('%05d',seqID);
    seqLen = 300;
    for i = 1:seqLen
        sequence.dataset{i} =  fullfile(seqPath, sprintf('img%03d%03d.jpg', seqID, i));
    end
    sequence.seqName = seqName;
    img = imread(sequence.dataset{1});
    [sequence.imgHeight, sequence.imgWidth, ~] = size(img);
    % nms processing
    detections = [];
    for i = 1:seqLen
        det = load(fullfile(detPath, sprintf('img%03d%03d_loc.txt',seqID,i)));
        if(size(det,2) ~= 3)
            det = det';       
        end
        if(~isempty(det))
            idx = det(:,3) > 0;
            det = det(idx,:);  
        end
        numdet = size(det,1);        
        if(numdet>0)
            curdet = [repmat([i, -1], [numdet, 1]), det(:,1)-10, det(:,2)-10, repmat([20, 20], [numdet, 1]), repmat([1, 1, 0, 0], [numdet, 1]), det(:,4:end)];
            detections = cat(1, detections, curdet);
        end
    end
    % add the tracker path
    cd(['./trackers/' trackerName]);
    addpath(genpath('.'));
    newres = [];
    % run the tracker
    if(~isempty(detections))
        [res, runTime] = run_tracker(seqLen, detections);
        objID = unique(res(:,2));
        newres = [];
        count = 0;
        for k = 1:length(objID)
            idx = res(:,2) == objID(k);
            if(nnz(idx)>=45) % remove short tracklets less than 45 frames
                count = count + 1;
                curres = res(idx,:);
                curres(:,2) = count;
                newres = cat(1, newres, curres);
            end
        end
        newres(:,8) = 1;
        speed = speed + runTime;                  
    end
    % remove the toolbox path
    rmpath(genpath('.'));
    cd('../../');                 
      
    % calculate the length of all the sequences
    numfrs = numfrs + seqLen;
    % show the tracking results
    dlmwrite(fullfile(resPath, [seqName '_' trackerName '.txt']), newres);
    showResults(isSeqDisplay, newres, sequence);
end

% calculate the speed
speed = numfrs/speed;
disp(['Tracking completed. The runing speed of ' trackerName ' tracker is ' num2str(roundn(speed,-2)) 'fps.']);