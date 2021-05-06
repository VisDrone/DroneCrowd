function runTrackerDets(isSeqDisplay, gtPath, datasetPath, detPath, resPath, seqPath, trackerName)

createPath(resPath); % create the saving path of the tracking results
seqList = load(seqPath); % find the sequence list

% evaluate each sequence
for idSeq = 1:length(seqList)
    if(mod(idSeq,10)==0)
        disp(['tracking the sequence ' num2str(idSeq) '/' num2str(length(seqList)) '...']);
    end
    % load detections and sequence
    seqID = seqList(idSeq);
    seqName = sprintf('%05d',seqID);
    sequence.dataset = dir(fullfile(datasetPath, [seqName '/*.jpg']));
    sequence.seqPath = fullfile(datasetPath, seqName);
    sequence.seqName = seqName;
    img = imread(fullfile(datasetPath, seqName, sequence.dataset(1).name));
    [sequence.imgHeight, sequence.imgWidth, ~] = size(img);
    % nms processing
    detections = [];
    for i = 1:length(sequence.dataset)
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
            curdet = [repmat([i, -1], [numdet, 1]), det(:,1)-10, det(:,2)-10, repmat([20, 20], [numdet, 1]), det(:,3), repmat([-1, -1, -1], [numdet, 1])];
            detections = cat(1, detections, curdet);
        end
    end
    % show the tracking results
    dlmwrite(fullfile(resPath, [seqName '_det.txt']), detections);

end